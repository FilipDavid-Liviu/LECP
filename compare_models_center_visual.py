import joblib
import matplotlib
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import config
import os
import copy
from feature_engineering import read_tif
matplotlib.use('Agg')

WINDOW_SIZE = 1500
CENTER_SHIFT = {
    "left": 300,
    "up": -100
}

MODEL_A_FILE = "best_control3_model.joblib"
MODEL_B_FILE = "best_lecp_model.joblib"


def get_sliding_windows(image, window_size=3):
    pad = window_size // 2
    padded_image = np.pad(image, pad_width=pad, mode='reflect')
    windows = sliding_window_view(padded_image, window_shape=(window_size, window_size))
    return windows


def prepare_raster_features(crop_rbr, crop_pre_nbr, model_type):
    """
    Transforms raw raster crops into features.
    Crucially, we handle NaNs to avoid cloud-bias errors.
    """
    mask = np.isnan(crop_rbr) | np.isnan(crop_pre_nbr)

    # Fill NaNs with 0 only for the feature array, but we will mask them later
    input_rbr = np.nan_to_num(crop_rbr, nan=0.0)
    input_pre = np.nan_to_num(crop_pre_nbr, nan=0.0)

    H, W = input_rbr.shape

    if "control" in model_type.lower():
        flat_rbr = input_rbr.reshape(-1, 1)
        flat_pre = input_pre.reshape(-1, 1)
        features = np.hstack([flat_rbr, flat_pre])
    else:
        # LECP Feature Extraction (18 features)
        patches_pre = get_sliding_windows(input_pre, 3).reshape(H * W, 9)
        patches_rbr = get_sliding_windows(input_rbr, 3).reshape(H * W, 9)
        features = np.hstack([patches_pre, patches_rbr])

    return features, mask


def save_single_map(data, title, filename, vmin=0, vmax=0.8):
    plt.figure(figsize=(10, 10))
    cmap = copy.copy(plt.cm.RdYlGn)
    cmap.set_bad(color='lightgrey')

    plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(title, fontsize=15)
    plt.axis('off')
    plt.colorbar(shrink=0.8).set_label('NDVI')

    out_path = os.path.join(config.PLOT_DIR, filename)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    # 1. Load Pre-tuned Models
    print(f"--- Loading Tuned Models ---")
    model_control = joblib.load(os.path.join(config.TIF_DIR, MODEL_A_FILE))
    model_lecp = joblib.load(os.path.join(config.TIF_DIR, MODEL_B_FILE))

    print("--- Loading & Cropping Data ---")
    pre_nbr = read_tif("PreFire_NBR.tif")
    post_nbr = read_tif("PostFire_NBR.tif")
    rec_ndvi = read_tif("Recovery_NDVI.tif")

    rbr_full = (pre_nbr - post_nbr) / (pre_nbr + 1.001)

    # Calculate crop area around the fire center
    valid_burns = np.argwhere(rbr_full > 0.2)
    center_r, center_c = valid_burns.mean(axis=0).astype(int) if len(valid_burns) > 0 else \
        (pre_nbr.shape[0] // 2, pre_nbr.shape[1] // 2)

    center_c = max(0, center_c - CENTER_SHIFT["left"])
    center_r = max(0, center_r - CENTER_SHIFT["up"])

    half = WINDOW_SIZE // 2
    r_s, r_e = max(0, center_r - half), min(pre_nbr.shape[0], center_r + half)
    c_s, c_e = max(0, center_c - half), min(pre_nbr.shape[1], center_c + half)

    crop_rbr = rbr_full[r_s:r_e, c_s:c_e]
    crop_pre = pre_nbr[r_s:r_e, c_s:c_e]
    crop_actual = rec_ndvi[r_s:r_e, c_s:c_e]
    H, W = crop_rbr.shape

    # 2. Predict with Masking
    print("--- Generating Predictions ---")
    # Prepare features and retrieve the NaN mask
    feats_cntrl, mask_cntrl = prepare_raster_features(crop_rbr, crop_pre, "control")
    pred_control = model_control.predict(feats_cntrl).reshape(H, W)

    feats_lecp, mask_lecp = prepare_raster_features(crop_rbr, crop_pre, "lecp")
    pred_lecp = model_lecp.predict(feats_lecp).reshape(H, W)

    # 3. Apply Strict Masking for Visuals
    # Hide clouds AND unburnt areas (RBR < 0.1) from the final map
    final_mask = mask_lecp | (crop_rbr < 0.1) | np.isnan(crop_actual)

    valid_idx = ~final_mask.flatten()
    y_true = crop_actual.flatten()[valid_idx]
    y_pred_ctrl = pred_control.flatten()[valid_idx]
    y_pred_lecp = pred_lecp.flatten()[valid_idx]

    print("\n" + "=" * 50)
    print(" NUMERIC RESULTS FOR VISUALIZED WINDOW")
    print("=" * 50)
    print(f"{'Metric':<10} | {'Control 3':<12} | {'LECP':<12} | {'Delta'}")
    print("-" * 50)

    for metric_name, func in [("R2", r2_score), ("MSE", mean_squared_error)]:
        val_ctrl = func(y_true, y_pred_ctrl)
        val_lecp = func(y_true, y_pred_lecp)
        delta = val_lecp - val_ctrl if metric_name == "R2" else val_ctrl - val_lecp
        print(f"{metric_name:<10} | {val_ctrl:.5f}      | {val_lecp:.5f}      | {delta:+.5f}")

    print(
        f"{'RMSE':<10} | {np.sqrt(mean_squared_error(y_true, y_pred_ctrl)):.5f}      | {np.sqrt(mean_squared_error(y_true, y_pred_lecp)):.5f}      |")
    print("=" * 50 + "\n")

    pred_control[final_mask] = np.nan
    pred_lecp[final_mask] = np.nan
    actual_masked = crop_actual.copy()
    actual_masked[final_mask] = np.nan

    # 4. Save Final Maps
    print("--- Saving Visual Comparison ---")
    save_single_map(actual_masked, "Ground Truth Recovery", "Visual_1_Actual.png")
    save_single_map(pred_control, "Control (Pixel-Only) Prediction", "Visual_2_Control.png")
    save_single_map(pred_lecp, "LECP (Spatial Context) Prediction", "Visual_3_LECP.png")

    # Improvement Map (Abs Error Control - Abs Error LECP)
    improvement = np.abs(pred_control - actual_masked) - np.abs(pred_lecp - actual_masked)

    plt.figure(figsize=(10, 10))
    cmap_imp = copy.copy(plt.cm.PiYG)
    cmap_imp.set_bad(color='lightgrey')
    plt.imshow(improvement, cmap=cmap_imp, vmin=-0.1, vmax=0.1)
    plt.title("LECP Improvement Map\n(Green = LECP is more accurate)", fontsize=14)
    plt.colorbar(shrink=0.8)
    plt.savefig(os.path.join(config.PLOT_DIR, "Visual_4_Improvement.png"), dpi=300)
    print("Visual analysis complete.")


if __name__ == "__main__":
    main()

