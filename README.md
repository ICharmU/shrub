# Shrub Detection Pipeline - Submission Runner

This directory contains a reproducible Jupyter notebook that executes the complete shrub detection ML pipeline end-to-end: **Modeling → Postprocessing → Evaluation**.

## Quick Start

### 1. Setup Environment

```bash
# Create conda environment
conda create --name shrub_env python=3.10
conda activate shrub_env

# Navigate to this directory
cd shrub

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Notebook

```bash
jupyter notebook main.ipynb
```

Crucial:
The data is expensive to fetch and generate so you can execute the sections in order, or copy over provided gdrive credentials which are onyl valid for this remote artifacts folder linked here: https://drive.google.com/drive/folders/1F9AtiUfx_z48tQIkNbDpzZUpH6R5uoCN?usp=drive_link

Execute sections in order but to just see it all in action you can skip straight to the final large modeling/postprocessing section provided the data exists locally.

## Pipeline Overview

### 1. Modeling Pipeline

- **Input**: Feature rasters (NAIP, ALS, 3DEP) for a single site
- **Output**: Probability predictions (`probability_raster.tif`)
- **Config**:
  - `feature_profile`: `naip` | `naip_als` | `naip_als_3dep` | `all_sources`
  - `model_family`: `pixel_logreg` | `small_cnn` | `custom_unet`
  - `fold_strategy`: `leave_one_site_out` (site-aware cross-validation)

### 2. Postprocessing Pipeline

- **Input**: Probability predictions
- **Output**:
  - `binary_raster.tif` — Binary shrub/background mask
  - `object_id_raster.tif` — Instance-labeled raster
  - `predicted_objects.csv` — Detected shrub objects with properties (area, centroid, etc.)
  - `site_summary.json` — Aggregate statistics (cover fraction, object count, etc.)
- **Config**:
  - `threshold_mode`: `fixed` (0.5) or `validation_tuned`
  - `cleanup_mode`: `none` | `standard` | `aggressive`
  - `split_mode`: `connected_components` | `watershed`

### 3. Evaluation Pipeline

- **Input**: Predictions + Label masks (if available)
- **Output**:
  - `site_metrics.json` — Per-site metrics:
    - **Raster Score**: avg(Dice, IoU) on pixel-level agreement
    - **Object Score**: avg(F1, count_agreement) from object matching
    - **Multi-Res Score**: mean IoU across 2m, 5m, 10m resolutions
    - **Site Summary Score**: agreement on aggregate statistics
    - **Composite Score**: weighted average (0.35 raster + 0.30 object + 0.15 multires + 0.20 site_summary)
  - `composite_scores.csv` — Aggregated performance across sites
- **Config**:
  - `object_iou_match_threshold`: Minimum IoU to consider object match (default 0.2)

## Configuration

Modify parameters in **Cell 4** to control pipeline behavior:

```python
# Choose a site
TEST_SITE = "sedgwick"  # or: dl-bliss, shaver-lake, etc.

# Modeling config
modeling_cfg = ModelingPipelineConfig(
    feature_profile="naip_als",      # Which features to use
    model_family="pixel_logreg",     # Which model architecture
    fold_strategy="leave_one_site_out",  # Cross-validation strategy
    batch_size=64,
    max_epochs=10,
    learning_rate=0.001,
    use_confidence_weights=True,     # Weight loss by label confidence
)

# Postprocessing config
postprocessing_cfg = PostprocessingPipelineConfig(
    threshold_mode="validation_tuned",  # How to threshold probability
    cleanup_mode="standard",            # Morphological cleanup
    split_mode="watershed",             # How to separate touching objects
)

# Evaluation config
evaluation_cfg = EvaluationPipelineConfig(
    object_iou_match_threshold=0.2,
)
```

## Available Sites

```
- calaveras-big-trees
- dl-bliss
- independence-lake
- pacific-union-college
- sedgwick (default)
- shaver-lake
```

## Output Directories

Pipeline outputs are organized by stage:

```
{cfg.output.root}/
├── modeling/pipeline_runs/submission_run/
│   └── {config_signature}/
│       ├── probability_raster.tif
│       ├── binary_raster.tif
│       └── ...
├── postprocessing/pipeline_runs/submission_run/
│   └── {config_signature}/
│       ├── binary_raster.tif
│       ├── object_id_raster.tif
│       ├── predicted_objects.csv
│       └── site_summary.json
└── evaluation/pipeline_runs/submission_run/
    └── {config_signature}/
        ├── site_metrics.json
        └── composite_scores.csv
```

Default `cfg.output.root` is in `Final/artifacts/`.

## Metric Explanations

### Raster Score (35% weight)

- **Dice**: `2 * intersection / (label_pixels + pred_pixels)` — Overlap ratio
- **IoU**: `intersection / union` — Jaccard index
- **Formula**: `(Dice + IoU) / 2`

### Object Score (30% weight)

- **Object F1**: Precision and recall from greedy IoU-based matching (IoU ≥ 0.2)
- **Count Agreement**: `1 - |label_count - pred_count| / label_count`
- **Formula**: `(F1 + Count_Agreement) / 2`

### Multi-Res Score (15% weight)

- **Process**: Downsample to 2m, 5m, 10m resolutions and compute IoU
- **Formula**: `mean(IoU@2m, IoU@5m, IoU@10m)`
- **Purpose**: Tests if model generalizes to coarser scales

### Site Summary Score (20% weight)

- **Metrics**: Agreement on cover_fraction, count_density, mean_object_area, median_object_area
- **Formula**: `mean(1 - |label_val - pred_val| / max(|label_val|, 1.0))` per metric
- **Purpose**: Tests if aggregate statistics are realistic

### Composite Score

```
composite = (
    0.35 * raster_score +
    0.30 * object_score +
    0.15 * multires_score +
    0.20 * site_summary_score
)
```

**Interpretation**:

- **0.0–0.3**: Poor (unacceptable)
- **0.3–0.5**: Weak generalization
- **0.5–0.7**: Moderate quality
- **0.7–0.85**: Good model
- **0.85–1.0**: Excellent (near-perfect)

## Dependencies

See `requirements.txt` for full list. Key packages:

- **ML**: torch, torchvision, scikit-learn, captum
- **Geospatial**: rasterio, scikit-image, scipy
- **I/O**: pandas, numpy, gdown (for Google Drive downloads)
- **Config**: pydrive2, PyWavelets, opencv-python-headless

## Troubleshooting

### Conda environment not found

```bash
conda env list
conda activate shrub_env  # or your env name
```

### Module not found errors

Ensure you're running in the right conda environment:

```bash
which python  # Should show .../shrub_env/...
```

### GPU not detected

Models will fall back to CPU automatically. To use GPU:

- Install CUDA: https://developer.nvidia.com/cuda-downloads
- Reinstall PyTorch with CUDA support:
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  ```

### Out of memory

Reduce `batch_size` in Cell 4:

```python
modeling_cfg.batch_size = 32  # or 16
```

### Metrics not showing

Evaluation metrics will be None if label data isn't available for the test site. This is expected in standalone prediction mode.

## Contact

For issues or questions, refer to the main project repository: https://github.com/ICharmU/shrub
