import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from sklearn.ensemble import RandomForestRegressor
from numpy.lib.stride_tricks import sliding_window_view
import config
import os

# --- SETTINGS ---
WINDOW_SIZE = 2000
SEED = 42

# --- CHOOSE YOUR FIGHTERS ---
# Available: "Standard" (RBR Only), "Control3" (RBR+PreNBR), "LECP" (Patches)
MODEL_A_NAME = "Control3"
MODEL_B_NAME = "LECP"


def read_tif(filename):
    path = os.path.join(config.OUTPUT_DIR, filename)
    with rasterio.open(path) as src:
        data = src.read(1).astype('float32')
        data[np.isinf(data)] = np.nan
        return data, src.profile


def get_sliding_windows(image, window_size=3):
    pad = window_size // 2
    padded_image = np.pad(image, pad_width=pad, mode='reflect')
    windows = sliding_window_view(padded_image, window_shape=(window_size, window_size))
    return windows


# --- 1. ABSTRACT DATA LOADER (For Training) ---
def get_training_data(df, model_type):
    """Returns X and y for the specific model type."""
    y = df['Target_RecoveryNDVI']

    if model_type == "Standard":
        # Control 1: RBR Only
        X = df[['Control_RBR']]

    elif model_type == "Control3":
        # Control 3: RBR + PreNBR (Pixel Level)
        X = df[['Control_RBR', 'Control_PreNBR']]

    elif model_type == "LECP":
        # Control 4: Full Patches
        # Drop known control columns to leave only patch features
        drop_cols = ['Target_RecoveryNDVI', 'Control_RBR', 'Control_PreNBR', 'Control_PostNBR']
        X = df.drop(columns=[c for c in df.columns if c in drop_cols])

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return X, y


# --- 2. ABSTRACT TRAINER ---
def train_specific_model(model_type):
    print(f"--- Training {model_type} ---")
    data_path = os.path.join(config.BASE_DIR, "training_dataset.csv")
    df = pd.read_csv(data_path)

    X, y = get_training_data(df, model_type)

    # Use consistent RF settings
    rf = RandomForestRegressor(n_estimators=50, n_jobs=-1, max_depth=10, random_state=SEED)
    rf.fit(X, y)
    return rf


# --- 3. ABSTRACT RASTER FEATURE PREPARATION ---
def prepare_raster_features(crop_rbr, crop_pre_nbr, model_type):
    """
    Transforms raw raster crops into the correct shape (N_samples, N_features)
    expected by the specific model type.
    """
    # Handle NaNs (fill with 0 for prediction, masked later)
    input_rbr = np.nan_to_num(crop_rbr, nan=0.0)
    input_pre = np.nan_to_num(crop_pre_nbr, nan=0.0)

    H, W = input_rbr.shape

    if model_type == "Standard":
        # Needs (N, 1) -> Just RBR
        return input_rbr.reshape(-1, 1)

    elif model_type == "Control3":
        # Needs (N, 2) -> RBR col, PreNBR col
        flat_rbr = input_rbr.reshape(-1, 1)
        flat_pre = input_pre.reshape(-1, 1)
        return np.hstack([flat_rbr, flat_pre])

    elif model_type == "LECP":
        # Needs (N, 18) -> 9 Pre Patches + 9 RBR Patches
        patches_pre = get_sliding_windows(input_pre, 3)
        patches_rbr = get_sliding_windows(input_rbr, 3)

        flat_pre_patches = patches_pre.reshape(H * W, 9)
        flat_rbr_patches = patches_rbr.reshape(H * W, 9)
        return np.hstack([flat_pre_patches, flat_rbr_patches])


def main():
    # 1. Train the two chosen models
    model_a = train_specific_model(MODEL_A_NAME)
    model_b = train_specific_model(MODEL_B_NAME)

    # 2. Load Full Rasters
    print("--- Loading Full Satellite Images ---")
    pre_nbr, profile = read_tif("PreFire_NBR.tif")
    rec_ndvi, _ = read_tif("Recovery_NDVI.tif")
    post_nbr, _ = read_tif("PostFire_NBR.tif")

    dnbr = pre_nbr - post_nbr
    rbr_full = dnbr / (pre_nbr + 1.001)

    # 3. Locate Fire Center
    print("--- Locating Fire Center ---")
    valid_burns = np.argwhere(rbr_full > 0.2)

    if len(valid_burns) == 0:
        print("Error: No fire found! Using image center.")
        center_r, center_c = pre_nbr.shape[0] // 2, pre_nbr.shape[1] // 2
    else:
        center_r, center_c = valid_burns.mean(axis=0).astype(int)

    # Crop Window
    half = WINDOW_SIZE // 2
    r_start = max(0, center_r - half)
    r_end = min(pre_nbr.shape[0], center_r + half)
    c_start = max(0, center_c - half)
    c_end = min(pre_nbr.shape[1], center_c + half)

    print(f"Cropping Window: {r_end - r_start}x{c_end - c_start}...")
    crop_pre_nbr = pre_nbr[r_start:r_end, c_start:c_end]
    crop_rbr = rbr_full[r_start:r_end, c_start:c_end]
    crop_actual = rec_ndvi[r_start:r_end, c_start:c_end]

    H, W = crop_rbr.shape

    # 4. Prepare Features & Predict
    print(f"--- Predicting with {MODEL_A_NAME} ---")
    feats_a = prepare_raster_features(crop_rbr, crop_pre_nbr, MODEL_A_NAME)
    pred_a = model_a.predict(feats_a).reshape(H, W)

    print(f"--- Predicting with {MODEL_B_NAME} ---")
    feats_b = prepare_raster_features(crop_rbr, crop_pre_nbr, MODEL_B_NAME)
    pred_b = model_b.predict(feats_b).reshape(H, W)

    # 5. Calculate Difference
    mask = (crop_rbr < 0.1)

    err_a = np.abs(pred_a - crop_actual)
    err_b = np.abs(pred_b - crop_actual)

    # Positive Value = Model B (LECP) has lower error
    improvement_map = err_a - err_b

    # Apply Mask
    improvement_map[mask] = np.nan
    pred_b[mask] = np.nan

    # 6. Visualization
    print("--- Saving Comparison Map ---")
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title(f"{MODEL_A_NAME} Error\n(Darker = Worse)")
    plt.imshow(err_a, cmap='Greys', vmin=0, vmax=0.4)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title(f"{MODEL_B_NAME} Error\n(Darker = Worse)")
    plt.imshow(err_b, cmap='Greys', vmin=0, vmax=0.4)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title(f"Improvement: {MODEL_B_NAME} vs {MODEL_A_NAME}\n(Green = {MODEL_B_NAME} Wins)")
    # PiYG: Pink (Negative/Model A wins) <-> Green (Positive/Model B wins)
    plt.imshow(improvement_map, cmap='PiYG', vmin=-0.15, vmax=0.15)
    plt.colorbar(label="Error Reduction")
    plt.axis('off')

    plt.tight_layout()
    out_file = os.path.join(config.BASE_DIR, f"map_compare_{MODEL_A_NAME}_vs_{MODEL_B_NAME}.png")
    plt.savefig(out_file, dpi=300)
    print(f"Map saved to: {out_file}")


if __name__ == "__main__":
    main()
