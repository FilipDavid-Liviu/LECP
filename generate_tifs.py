import os
import numpy as np
import rasterio
import config
import utils


def calculate_normalized_index(b1, b2):
    """Generic function for (B1 - B2) / (B1 + B2)"""
    np.seterr(divide='ignore', invalid='ignore')
    numerator = b1 - b2
    denominator = b1 + b2
    index = numerator / denominator
    index[denominator == 0] = np.nan
    return index


def process_step(step_name, bands):
    print(f"--- Processing {step_name} ---")

    # 1. Build full file paths
    p_b03 = os.path.join(config.BASE_DIR, bands["B03"])
    p_b04 = os.path.join(config.BASE_DIR, bands["B04"])
    p_b08 = os.path.join(config.BASE_DIR, bands["B08"])
    p_b12 = os.path.join(config.BASE_DIR, bands["B12"])
    p_scl = os.path.join(config.BASE_DIR, bands["SCL"])

    # 2. Read Basic Bands (10m)
    with rasterio.open(p_b04) as src:
        b04 = src.read(1).astype('float32')
        profile = src.profile

    with rasterio.open(p_b03) as src:
        b03 = src.read(1).astype('float32')

    with rasterio.open(p_b08) as src:
        b08 = src.read(1).astype('float32')

    # 3. Resample 20m bands to 10m
    print("Upsampling B12 and SCL to 10m...")
    b12, _ = utils.read_and_resample(p_b04, p_b12)
    scl = utils.read_scl_resampled(p_b04, p_scl)

    # 4. Mask Clouds
    print("Masking clouds...")
    b03 = utils.apply_mask(b03, scl)
    b04 = utils.apply_mask(b04, scl)
    b08 = utils.apply_mask(b08, scl)
    b12 = utils.apply_mask(b12, scl)

    # 6. Apply Water Mask to all bands
    ndwi = calculate_normalized_index(b03, b08)
    print("Applying Water Mask (NDWI > 0.0)...")
    b04 = utils.apply_water_mask(b04, ndwi, threshold=0.0)
    b08 = utils.apply_water_mask(b08, ndwi, threshold=0.0)
    b12 = utils.apply_water_mask(b12, ndwi, threshold=0.0)

    # 7. Calculate Indices
    # Flatten b12 dimension if necessary (it comes out (1, H, W))
    if b12.ndim == 3: b12 = b12[0]
    print("Calculating final NBR and NDVI...")
    nbr = calculate_normalized_index(b08, b12)
    ndvi = calculate_normalized_index(b08, b04)

    # 8. Save
    utils.save_tif(os.path.join(config.TIF_DIR, f"{step_name}_NBR.tif"), [nbr], profile)
    utils.save_tif(os.path.join(config.TIF_DIR, f"{step_name}_NDVI.tif"), [ndvi], profile)


if __name__ == "__main__":
    process_step("PreFire", config.DATA_FILES["pre_fire"])
    process_step("PostFire", config.DATA_FILES["post_fire"])
    process_step("Recovery", config.DATA_FILES["recovery"])

    print("\nProcessing Complete. You now have the raw ingredients for the LECP.")
