import os
import numpy as np
import rasterio
import config
import utils


def calculate_normalized_index(b1, b2):
    """Generic function for (B1 - B2) / (B1 + B2)"""
    # Ignore division by zero warnings
    np.seterr(divide='ignore', invalid='ignore')

    numerator = b1 - b2
    denominator = b1 + b2

    index = numerator / denominator

    # Clean up infinity values
    index[denominator == 0] = np.nan
    return index


def process_step(step_name, bands):
    print(f"--- Processing {step_name} ---")

    # 1. Build full file paths
    p_b04 = os.path.join(config.BASE_DIR, bands["B04"])
    p_b08 = os.path.join(config.BASE_DIR, bands["B08"])
    p_b12 = os.path.join(config.BASE_DIR, bands["B12"])
    p_scl = os.path.join(config.BASE_DIR, bands["SCL"])

    # 2. Read Basic Bands (10m)
    with rasterio.open(p_b04) as src:
        b04 = src.read(1).astype('float32')
        profile = src.profile

    with rasterio.open(p_b08) as src:
        b08 = src.read(1).astype('float32')

    # 3. Resample 20m bands to 10m
    print("Upsampling B12 and SCL to 10m...")
    b12, _ = utils.read_and_resample(p_b04, p_b12)
    scl = utils.read_scl_resampled(p_b04, p_scl)

    # 4. Mask Clouds
    print("Masking clouds...")
    b04 = utils.apply_mask(b04, scl)
    b08 = utils.apply_mask(b08, scl)
    b12 = utils.apply_mask(b12, scl)  # b12 is already 3D from utils, select band 0 if needed

    # 5. Calculate Indices
    # NBR = (NIR - SWIR) / (NIR + SWIR) -> (B08 - B12)
    # NDVI = (NIR - Red) / (NIR + Red) -> (B08 - B04)

    # Flatten b12 dimension if necessary (it comes out (1, H, W))
    if b12.ndim == 3: b12 = b12[0]
    if scl.ndim == 3: scl = scl[0]

    print("Calculating indices...")
    nbr = calculate_normalized_index(b08, b12)
    ndvi = calculate_normalized_index(b08, b04)

    # 6. Save
    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)

    utils.save_tif(os.path.join(config.OUTPUT_DIR, f"{step_name}_NBR.tif"), [nbr], profile)
    utils.save_tif(os.path.join(config.OUTPUT_DIR, f"{step_name}_NDVI.tif"), [ndvi], profile)


if __name__ == "__main__":
    # Run Pre-Fire
    process_step("PreFire", config.DATA_FILES["pre_fire"])

    # Run Post-Fire
    process_step("PostFire", config.DATA_FILES["post_fire"])

    # Run Recovery
    process_step("Recovery", config.DATA_FILES["recovery"])

    print("\nProcessing Complete. You now have the raw ingredients for the LECP.")
