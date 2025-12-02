import rasterio
from rasterio.enums import Resampling
import numpy as np


def read_and_resample(path_10m_ref, path_target):
    """
    Reads a target band (e.g., 20m B12) and resamples it to match
    the resolution and shape of a reference band (e.g., 10m B04).
    """
    # Open the 10m file just to get the target dimensions (height/width)
    with rasterio.open(path_10m_ref) as ref:
        out_shape = (ref.count, ref.height, ref.width)
        profile = ref.profile

    # Open the target file (20m) and read it into the 10m dimensions
    with rasterio.open(path_target) as src:
        data = src.read(
            out_shape=out_shape,
            resampling=Resampling.bilinear  # Bilinear is best for continuous data like B12
        )
        # Scale the data if needed (Sentinel-2 JP2 is usually 0-10000 integers)
        data = data.astype('float32')

    return data[0], profile


def read_scl_resampled(path_10m_ref, path_scl):
    """
    Special reader for SCL. We use Nearest Neighbor resampling because
    SCL relies on specific integers (e.g., 8=Cloud), not averages.
    """
    with rasterio.open(path_10m_ref) as ref:
        out_shape = (ref.count, ref.height, ref.width)

    with rasterio.open(path_scl) as src:
        data = src.read(
            out_shape=out_shape,
            resampling=Resampling.nearest
        )
    return data[0]


def apply_mask(band_data, scl_data):
    """
    Sets pixel values to NaN if the SCL indicates clouds/shadows.
    """
    # SCL Classes: 0 (No Data), 1 (Defect), 3 (Shadow), 8-10 (Clouds)
    # Note: Class 6 is Water. We CAN use SCL=6 to mask water, but calculating NDWI manually
    # is often more accurate for river edges than the coarse SCL.
    bad_pixels = np.isin(scl_data, [0, 1, 3, 8, 9, 10])

    masked_data = band_data.copy()
    masked_data[bad_pixels] = np.nan
    return masked_data


def apply_water_mask(band_data, ndwi_data, threshold=0.0):
    """
    Sets pixel values to NaN if NDWI > threshold (meaning it is water).
    McFeeters NDWI: (Green - NIR) / (Green + NIR)
    """
    water_pixels = ndwi_data > threshold

    masked_data = band_data.copy()
    # Apply NaN to water areas
    masked_data[water_pixels] = np.nan
    return masked_data


def save_tif(filename, data, profile):
    """Saves the result as a GeoTIFF"""
    # Update profile for float32 data (since we calculated indices)
    profile.update(
        dtype=rasterio.float32,
        count=1,
        driver='GTiff',
        compress='lzw'
    )
    with rasterio.open(filename, 'w', **profile) as dst:
        dst.write(data[0], 1)
    print(f"Saved: {filename}")
