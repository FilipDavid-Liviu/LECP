# LECP: Local Ecological Context Patch for Fire Recovery Analysis
This repository contains a remote sensing pipeline designed to analyze post-fire vegetation recovery. It tests the hypothesis that including local spatial context (neighboring pixels) significantly improves the prediction of recovery outcomes compared to using single-pixel severity metrics alone.

The pipeline processes Sentinel-2 satellite imagery to generate spectral indices, filters for valid fire data, trains machine learning models, and evaluates performance using both statistical metrics and spatial visualization.

## Workflow Overview
### 1. Data Processing and Index Generation
The first stage transforms raw multi-spectral satellite imagery into analysis-ready spectral indices.
- Input: Raw Sentinel-2 bands (10m and 20m resolution) for three distinct timepoints: Pre-Fire, Post-Fire, and Recovery.
- Resampling: Higher resolution bands are used as the baseline; lower resolution bands are upsampled to match.
- Masking:
  - Cloud Masking: Pixels identified as clouds or shadows by the SCL (Scene Classification Layer) are removed.
  - Water Masking: A NDWI (Normalized Difference Water Index) is calculated. Pixels exceeding the threshold are masked out to prevent rivers and lakes from skewing burn severity statistics.
- Index Calculation:
  - NBR (Normalized Burn Ratio): Calculated for Pre-fire and Post-fire images.
  - NDVI (Normalized Difference Vegetation Index): Calculated for the Recovery image to serve as the ground truth target.

### 2. Feature Engineering and Dataset Creation
This stage converts the processed raster images into a structured tabular dataset suitable for machine learning.
- Metric Calculation:
  - dNBR: Difference between Pre-fire and Post-fire NBR.
  - RBR (Relativized Burn Ratio): A relativistic metric derived from dNBR that accounts for pre-fire vegetation density.
- Spatial Patch Extraction: Instead of treating pixels in isolation, the system extracts 3x3 pixel neighborhoods around every sample point. This captures the local ecological context.
- Filtering: The dataset is filtered to exclude non-vegetated areas, water bodies, and unburned pixels (based on a severity threshold).
- Sampling: A randomized subset of valid pixels is selected to create a balanced training dataset.

### 3. Model Training (Global Validation)
This step validates the hypothesis using standard machine learning cross-validation techniques.
- Model Architectures: Four distinct Random Forest configurations are trained to isolate specific variables:
  - Control 1: Uses only Burn Severity (RBR).
  - Control 2: Uses only Pre-fire Health (PreNBR).
  - Control 3: Uses a combination of Severity and Pre-fire Health (Single Pixel).
  - LECP (Experimental): Uses the 3x3 spatial patches for both Severity and Pre-fire Health.
- Evaluation: The dataset is split into training and testing sets (random holdout). Metrics such as RMSE, R², and MAE are calculated to compare the predictive power of the spatial approach against the control models.

### 4. Spatial Inference and Visualization
The final stage moves beyond statistical metrics to evaluate how the models perform on a continuous landscape.
- Scene Generation: A specific fire event is reconstructed from the raster data.
- Inference: The trained models generate predictions for every pixel in the scene, reconstructing a full "Predicted Recovery" map.
- Strict Validation: A rigorous mask is applied during this phase to strictly exclude water and low-severity noise, ensuring the evaluation focuses solely on significant fire effects.
- Comparative Analysis:
  - Visual Maps: Side-by-side comparisons of Ground Truth vs. Model Predictions.
  - Error Mapping: A difference map is generated to visualize exactly where the spatial model outperforms the single-pixel controls (e.g., along fire perimeters or in heterogeneous terrain).