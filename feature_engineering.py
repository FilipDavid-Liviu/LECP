import os
import numpy as np
import rasterio
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
import config

# --- CONFIGURATION ---
SAMPLE_SIZE = 50000
BURN_THRESHOLD = 0.1
BLOCK_SIZE = 50


def read_tif(filename):
    path = os.path.join(config.TIF_DIR, filename)
    with rasterio.open(path) as src:
        data = src.read(1).astype('float32')
        data[np.isinf(data)] = np.nan
        return data


if __name__ == "__main__":
    print("--- Loading TIF Data ---")
    pre_nbr = read_tif("PreFire_NBR.tif")
    post_nbr = read_tif("PostFire_NBR.tif")
    rec_ndvi = read_tif("Recovery_NDVI.tif")

    print("--- Calculating Derived Metrics ---")

    dnbr = pre_nbr - post_nbr
    rbr = dnbr / (pre_nbr + 1.001)

    target_absolute = rec_ndvi

    print(f"--- Identifying Valid Pixels (Threshold > {BURN_THRESHOLD}) ---")
    valid_mask = (
            ~np.isnan(pre_nbr) &
            ~np.isnan(rbr) &
            ~np.isnan(target_absolute) &
            (rbr > BURN_THRESHOLD) &
            (target_absolute > -1.0) & (target_absolute < 1.0)
    )

    valid_coords = np.argwhere(valid_mask)
    print(f"Found {len(valid_coords)} valid pixels.")

    if len(valid_coords) == 0:
        print("ERROR: No valid pixels found.")
        exit(1)

    # Randomly Sample
    if len(valid_coords) > SAMPLE_SIZE:
        print(f"Subsampling to {SAMPLE_SIZE} points...")
        indices = np.random.choice(len(valid_coords), SAMPLE_SIZE, replace=False)
        sample_coords = valid_coords[indices]
    else:
        sample_coords = valid_coords

    print("--- Extracting Patches ---")

    # Pad images for 3x3 extraction
    pad_pre_nbr = np.pad(pre_nbr, pad_width=1, mode='reflect')
    pad_rbr = np.pad(rbr, pad_width=1, mode='reflect')

    lecp_pre_features = np.zeros((len(sample_coords), 9), dtype=np.float32)
    lecp_rbr_features = np.zeros((len(sample_coords), 9), dtype=np.float32)

    control_rbr_list = []
    control_prenbr_list = []
    targets = []
    block_ids = []

    for i, (r, c) in enumerate(sample_coords):
        # 1. Spatial Block Assignment
        block_r = r // BLOCK_SIZE
        block_c = c // BLOCK_SIZE
        block_ids.append(f"B_{block_r}_{block_c}")

        # 2. Patch Extraction
        pr, pc = r + 1, c + 1
        w_pre = pad_pre_nbr[pr - 1:pr + 2, pc - 1:pc + 2]
        w_rbr = pad_rbr[pr - 1:pr + 2, pc - 1:pc + 2]

        lecp_pre_features[i, :] = w_pre.flatten()
        lecp_rbr_features[i, :] = w_rbr.flatten()

        control_rbr_list.append(rbr[r, c])
        control_prenbr_list.append(pre_nbr[r, c])
        targets.append(target_absolute[r, c])

    print("--- Building DataFrame ---")
    df = pd.DataFrame()
    df['Target_RecoveryNDVI'] = targets
    df['Spatial_Block_ID'] = block_ids
    df['Control_RBR'] = control_rbr_list
    df['Control_PreNBR'] = control_prenbr_list

    for p_idx in range(9):
        df[f"PreNBR_P{p_idx}"] = lecp_pre_features[:, p_idx]
        df[f"RBR_P{p_idx}"] = lecp_rbr_features[:, p_idx]

    df = df.dropna()

    output_path = os.path.join(config.TIF_DIR, "training_dataset.csv")
    df.to_csv(output_path, index=False)
    print(f"Success! Dataset with Spatial Blocks saved to: {output_path}")
