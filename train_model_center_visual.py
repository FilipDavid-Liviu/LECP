import matplotlib
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import config
import os
import copy
from feature_engineering import read_tif, get_sliding_windows
matplotlib.use('Agg')

# --- CONFIGURATION ---
WINDOW_SIZE = 2000
SEED = 42

MODEL_A_NAME = "Control3"
MODEL_B_NAME = "LECP"


# --- 1. DATA LOADER ---
def get_training_data(df, model_type):
    y = df['Target_RecoveryNDVI']

    if model_type == "Control1":
        X = df[['Control_RBR']]
    elif model_type == "Control2":
        X = df[['Control_PreNBR']]
    elif model_type == "Control3":
        X = df[['Control_RBR', 'Control_PreNBR']]
    elif model_type == "LECP":
        drop_cols = ['Target_RecoveryNDVI', 'Control_RBR', 'Control_PreNBR', 'Control_PostNBR']
        X = df.drop(columns=[c for c in df.columns if c in drop_cols])
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return X, y


# --- 2. TRAINER ---
def train_specific_model(model_type):
    print(f"--- Training {model_type} ---")
    data_path = os.path.join(config.TIF_DIR, "training_dataset.csv")
    df = pd.read_csv(data_path)
    X, y = get_training_data(df, model_type)

    rf = RandomForestRegressor(
        n_estimators=100,
        n_jobs=-1,
        max_depth=25,  # Restricts depth slightly to prevent pure memorization
        min_samples_leaf=4,  # Requires at least 4 pixels to agree before making a rule
        random_state=SEED
    )
    rf.fit(X, y)
    return rf


# --- 3. FEATURE PREPARATION ---
def prepare_raster_features(crop_rbr, crop_pre_nbr, model_type):
    """
    Transforms raw raster crops into the correct shape (N_samples, N_features)
    expected by the specific model type.
    """
    # Handle NaNs (fill with 0 for prediction, masked later)
    input_rbr = np.nan_to_num(crop_rbr, nan=0.0)
    input_pre = np.nan_to_num(crop_pre_nbr, nan=0.0)

    H, W = input_rbr.shape

    if model_type == "Control3":
        # Training data was: df[['Control_RBR', 'Control_PreNBR']]
        # So we must stack RBR then Pre
        flat_rbr = input_rbr.reshape(-1, 1)
        flat_pre = input_pre.reshape(-1, 1)
        return np.hstack([flat_rbr, flat_pre])

    elif model_type == "LECP":
        # 1. Get Sliding Windows (Shape: H, W, 3, 3)
        patches_pre = get_sliding_windows(input_pre, 3)
        patches_rbr = get_sliding_windows(input_rbr, 3)

        # 2. Flatten the 3x3 window into 9 features per pixel
        # Shape becomes: (N_pixels, 9)
        flat_pre_patches = patches_pre.reshape(H * W, 9)
        flat_rbr_patches = patches_rbr.reshape(H * W, 9)

        # --- THE FIX ---
        # The CSV was built interleaved: Pre_0, RBR_0, Pre_1, RBR_1...
        # We must replicate that structure.

        # Stack them along a new axis -> (N_pixels, 9, 2)
        # Axis 2, index 0 is Pre, index 1 is RBR
        stacked = np.stack([flat_pre_patches, flat_rbr_patches], axis=2)

        # Reshape to flatten the last two dimensions -> (N_pixels, 18)
        # This forces the order: Pre0, RBR0, Pre1, RBR1...
        return stacked.reshape(H * W, 18)


# --- 4. VISUALIZATION HELPER ---
def save_single_map(data, title, filename, vmin=0, vmax=0.8):
    """
    Saves a single generic map with the RdYlGn colormap.
    NaNs are rendered as Grey to represent Clouds/No-Data.
    """
    plt.figure(figsize=(10, 10))

    # Create a copy of the colormap to modify bad value color
    cmap = copy.copy(plt.cm.RdYlGn)
    cmap.set_bad(color='lightgrey')  # This makes NaNs/Clouds Grey

    plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(title, fontsize=15)
    plt.axis('off')

    # Add a colorbar
    cbar = plt.colorbar(shrink=0.8)
    cbar.set_label('NDVI Recovery')

    out_path = os.path.join(config.PLOT_DIR, filename)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def main():
    # 1. Train & Load
    model_control = train_specific_model(MODEL_A_NAME)
    model_lecp = train_specific_model(MODEL_B_NAME)

    print("--- Loading & Cropping Data ---")
    pre_nbr = read_tif("PreFire_NBR.tif")
    rec_ndvi = read_tif("Recovery_NDVI.tif")
    post_nbr = read_tif("PostFire_NBR.tif")

    dnbr = pre_nbr - post_nbr
    rbr_full = dnbr / (pre_nbr + 1.001)

    # Locate Fire Center
    valid_burns = np.argwhere(rbr_full > 0.5)
    if len(valid_burns) == 0:
        center_r, center_c = pre_nbr.shape[0] // 2, pre_nbr.shape[1] // 2
    else:
        center_r, center_c = valid_burns.mean(axis=0).astype(int)

    half = WINDOW_SIZE // 2
    r_start, r_end = max(0, center_r - half), min(pre_nbr.shape[0], center_r + half)
    c_start, c_end = max(0, center_c - half), min(pre_nbr.shape[1], center_c + half)

    crop_pre_nbr = pre_nbr[r_start:r_end, c_start:c_end]
    crop_rbr = rbr_full[r_start:r_end, c_start:c_end]
    crop_actual = rec_ndvi[r_start:r_end, c_start:c_end]
    H, W = crop_rbr.shape

    # 2. Predict
    print(f"--- Predicting... ---")
    feats_cntrl = prepare_raster_features(crop_rbr, crop_pre_nbr, MODEL_A_NAME)
    pred_control = model_control.predict(feats_cntrl).reshape(H, W)

    feats_lecp = prepare_raster_features(crop_rbr, crop_pre_nbr, MODEL_B_NAME)
    pred_lecp = model_lecp.predict(feats_lecp).reshape(H, W)

    # 3. Masking & Metric Calculation
    print("--- Calculating Metrics & Improvement ---")

    # Strict Mask: Clouds in Ground Truth OR Clouds in Pre-fire OR Unburnt areas
    strict_mask = np.isnan(crop_actual) | (crop_rbr < 0.1) | np.isnan(crop_pre_nbr)

    # Apply mask to copies for calculation
    pred_control_masked = pred_control.copy()
    pred_lecp_masked = pred_lecp.copy()
    actual_masked = crop_actual.copy()

    pred_control_masked[strict_mask] = np.nan
    pred_lecp_masked[strict_mask] = np.nan
    actual_masked[strict_mask] = np.nan

    # Flatten arrays and remove NaNs for Scikit-Learn metrics
    # (Scikit-learn cannot handle NaNs, so we must compress the arrays)
    valid_idx = ~np.isnan(actual_masked.flatten())

    y_true = actual_masked.flatten()[valid_idx]
    y_pred_ctrl = pred_control_masked.flatten()[valid_idx]
    y_pred_lecp = pred_lecp_masked.flatten()[valid_idx]

    # --- PRINT RESULTS TO CONSOLE ---
    rmse_ctrl = np.sqrt(mean_squared_error(y_true, y_pred_ctrl))
    rmse_lecp = np.sqrt(mean_squared_error(y_true, y_pred_lecp))
    r2_ctrl = r2_score(y_true, y_pred_ctrl)
    r2_lecp = r2_score(y_true, y_pred_lecp)

    print("\n" + "=" * 40)
    print(f" RESULTS SUMMARY (Pixels: {len(y_true)})")
    print("=" * 40)
    print(f"{'METRIC':<10} | {MODEL_A_NAME:<12} | {MODEL_B_NAME:<12} | {'DELTA':<10}")
    print("-" * 52)
    print(f"{'RMSE':<10} | {rmse_ctrl:.5f}       | {rmse_lecp:.5f}       | {rmse_ctrl - rmse_lecp:+.5f}")
    print(f"{'R2':<10}   | {r2_ctrl:.5f}       | {r2_lecp:.5f}       | {r2_lecp - r2_ctrl:+.5f}")
    print("=" * 40 + "\n")
    # --------------------------------

    # 4. Improvement Map Calculation
    # We use the masked versions we created above
    abs_err_control = np.abs(pred_control_masked - crop_actual)
    abs_err_lecp = np.abs(pred_lecp_masked - crop_actual)
    improvement_map = abs_err_control - abs_err_lecp

    # 5. Generate Images
    print("--- Saving Images ---")
    VMIN_NDVI, VMAX_NDVI = 0.0, 0.8

    # Save Images 1, 2, 3
    save_single_map(actual_masked, "Actual Recovery (Ground Truth)", "1_Actual_Recovery.png", VMIN_NDVI, VMAX_NDVI)
    save_single_map(pred_control_masked, f"Prediction: {MODEL_A_NAME}", "2_Control_Prediction.png", VMIN_NDVI,
                    VMAX_NDVI)
    save_single_map(pred_lecp_masked, f"Prediction: {MODEL_B_NAME}", "3_LECP_Prediction.png", VMIN_NDVI, VMAX_NDVI)

    # Save Image 4 (Comparison)
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    IMP_LIM = 0.15

    # Left: Prediction
    cmap_pred = copy.copy(plt.cm.RdYlGn)
    cmap_pred.set_bad(color='lightgrey')
    im1 = axes[0].imshow(pred_lecp_masked, cmap=cmap_pred, vmin=VMIN_NDVI, vmax=VMAX_NDVI)
    axes[0].set_title(f"Prediction: {MODEL_B_NAME}\nRMSE: {rmse_lecp:.4f}", fontsize=14, fontweight='bold')
    axes[0].axis('off')
    cbar1 = fig.colorbar(im1, ax=axes[0], shrink=0.7, orientation='horizontal', pad=0.05)
    cbar1.set_label("Predicted NDVI")

    # Right: Improvement
    cmap_imp = copy.copy(plt.cm.PiYG)
    cmap_imp.set_bad(color='lightgrey')
    im2 = axes[1].imshow(improvement_map, cmap=cmap_imp, vmin=-IMP_LIM, vmax=IMP_LIM)
    axes[1].set_title(f"Improvement over {MODEL_A_NAME}\n(RMSE Improvement: {rmse_ctrl - rmse_lecp:.4f})", fontsize=14,
                      fontweight='bold')
    axes[1].axis('off')
    cbar2 = fig.colorbar(im2, ax=axes[1], shrink=0.7, orientation='horizontal', pad=0.05)
    cbar2.set_label("Error Reduction (Green = Better)")

    plt.suptitle(f"Model Analysis: {MODEL_B_NAME} Performance", fontsize=20, y=0.95)
    out_file = os.path.join(config.PLOT_DIR, "4_LECP_Analysis.png")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Analysis saved to: {out_file}")


if __name__ == "__main__":
    main()
