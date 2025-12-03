# Tree Canopy Detection and Instance Segmentation Pipeline

A geospatial machine learning workflow for automated tree crown detection, segmentation, and evaluation using satellite imagery (DOP - Digital Orthophotos) and vector annotations.

## Overview

This project implements an end-to-end pipeline for:
- **Data Preparation**: Vegetation masking, imagery slicing, and annotation cleaning
- **Dataset Organization**: YOLO-compatible dataset creation with train/validation splits
- **Model Training**: Multiple YOLOv8 variants with custom loss functions
- **Model Evaluation**: Comparative analysis using RandCrowns metric and IoU-based metrics


## Workflow Pipeline

### Stage 1: Data Preparation

#### 00_create_vegetation_mask.ipynb
- Generates vegetation masks from satellite imagery
- Filters out non-vegetation areas (buildings, roads, water)
- Output: Binary vegetation masks (.tif)

#### 01_01_Mask_Data.ipynb
- Applies vegetation masks to satellite imagery
- Removes non-vegetation pixels
- Output: Masked satellite images (.tif)

#### 01_03_DOP_slicing.ipynb
- Slices large Digital Orthophoto (DOP) imagery into overlapping tiles
- Tile size: 100m × 100m (configurable)
- Overlap: 20m per side to avoid edge artifacts
- Output: Sliced tile images (.tif)

#### 01_05_Data_Cleaning.ipynb
- Validates and cleans vector annotations (GeoJSON)
- Removes invalid geometries and duplicate annotations
- Filters by coverage threshold (e.g., >70% vegetation)
- Output: Cleaned polygon annotations (.geojson)

#### 01_06_Define_Ignore_Class.ipynb
- Defines background/ignore classes for model training
- Handles cadaster intersections and building footprints
- Creates multi-class annotation scheme
- Output: Annotated tiles with class labels (.geojson)

### Stage 2: Dataset Preparation

#### 02_01_Inner_Tile_Cropping_Experimental.ipynb
- Removes overlapping regions from sliced tiles
- Detects spatial neighbors using intersection tests
- Crops 10m margins based on neighbor position
- Converts polygon annotations to YOLO instance segmentation format
- **Key Feature**: Spatial neighbor detection to avoid duplicate annotations
- Output: 
  - Inner tile images (non-overlapping) (.tif)
  - YOLO annotations (.txt) with normalized polygon coordinates

#### 02_02_Training_Data_Preparation.ipynb
- Organizes inner tiles and YOLO annotations into train/validation split (80/20)
- Creates YOLO-compatible directory structure
- Generates `dataset.yaml` configuration for YOLOv8 training
- **Key Feature**: Automatic shuffling and deterministic splitting
- Output:
  - `yolo_dataset/images/train/` and `/val/`
  - `yolo_dataset/labels/train/` and `/val/`
  - `dataset.yaml` (YOLO configuration file)

#### 03_01_Creating_training_and_validation_dataset.ipynb
- Converts satellite images and vector annotations to COCO format
- Filters images by coverage threshold (70%)
- Converts polygons to binary instance masks
- Extracts contours and creates COCO annotations with bounding boxes
- **Key Feature**: Polygon-to-mask conversion with Polygon and MultiPolygon support
- Output:
  - `train.json` (COCO format with segmentation polygons)
  - `val.json` (COCO format with segmentation polygons)

### Stage 3: Model Training

#### 03_01a_Model_training_default_ultarlitics.ipynb
- Train YOLOv8 instance segmentation with default ultralytics configuration
- **Configuration**: Default hyperparameters, BCE loss
- **Output**: 
  - Trained model weights (.pt)
  - Training metrics (metrics.json)
  - Results plots (loss, mAP, etc.)

#### 03_01b_Model_training_custom_ultarlitics.ipynb
- Train YOLOv8 with custom hyperparameters
- **Configuration**: Custom learning rate, batch size, augmentation
- **Output**: Trained model weights (.pt) with custom config

#### 03_01c_Model_training_custom_dice_loss_ultarlitics.ipynb
- Train YOLOv8 with custom Dice loss function
- **Configuration**: Dice loss for better polygon segmentation
- **Motivation**: Dice loss optimizes for overlap region (better for instance segmentation)
- **Output**: Trained model weights (.pt) with Dice loss

### Stage 4: Model Evaluation

#### 04_01_Evaluation_RandCrowns.ipynb
- Evaluate trained YOLOv8 models using RandCrowns metric
- Compares predictions against:
  - **Ground truth** (lidR high-quality annotations)
  - **Cadaster boundaries** (property cadastral data)
- **RandCrowns Metric**: Custom metric combining intersection area, buffer zones, and asymmetric weighting
- **Detection Metrics**: AP@0.5, AP@0.75, Recall at multiple IoU thresholds
- **Tree Classification**: Separate analysis for street trees vs. park trees
- **Overfitting Detection**: Compare train/validation AP curves
- **Output**:
  - RandCrowns summary statistics (CSV)
  - Performance plots (AP, Recall, RandCrowns distribution)
  - False positive analysis (GeoPackage export)

#### 04_02_Evaluation_DeepForest_and_Dalponte.ipynb
- Comparative evaluation of three tree crown detection models:
  - **YOLO**: Custom-trained YOLOv8 instance segmentation
  - **DeepForest**: Pre-trained deep learning model (Weinstein et al.)
  - **Dalponte/Silva**: Segmentation-based algorithm
- **Evaluation Metrics**:
  - RandCrowns (tree crown-specific metric)
  - Confusion matrix (TP, FP, FN) at IoU [0.25, 0.50, 0.75]
  - F1-Score, Precision, Recall
- **Output**:
  - Comparison DataFrame with all metrics
  - Side-by-side visual predictions
  - Saved predictions (GeoPackage for manual inspection)

## Key Concepts

### RandCrowns Metric
Custom evaluation metric optimized for tree crown detection:
- Combines intersection area with buffer zones (tolerance for positional error)
- Asymmetric weighting for different detection scenarios
- Range: [0, 1], where 1 = perfect detection
- Better suited for tree crown evaluation than standard IoU

### YOLO Format
Instance segmentation format used by YOLOv8:
- One `.txt` file per image
- Each line: `class_id x1 y1 x2 y2 ... xn yn`
- Coordinates normalized to [0, 1] range
- Supports polygon segmentation masks

### Spatial Neighbor Detection
Technique used in tile cropping:
- Identifies overlapping tiles using spatial intersection
- Automatically determines which margins to crop
- Avoids duplicate annotations in overlapping regions
- Margin: 10m per side (configurable)

## Processing Parameters

### Data Slicing
- Tile size: 100m × 100m
- Overlap: 20m per side
- Image resolution: 0.20m/pixel (0.40m optional)
- CRS: EPSG:25833 (UTM Zone 33N, Berlin region)

### Dataset Split
- Train/Validation: 80/20
- Coverage threshold: >70% vegetation
- Random seed: 42 (for reproducibility)

### Training
- Image resolution: 0.20m/pixel
- Batch size: 8-16 (configurable)
- Epochs: 100-200
- Loss functions: BCE (default) or Dice (custom)
- Optimizer: SGD or Adam

### Evaluation
- Confidence threshold: 0.1
- IoU thresholds: [0.25, 0.50, 0.75]
- RandCrowns parameters: alpha=0.7, omega=1.2, gamma=3.0

## Input Data Requirements

- **Satellite Imagery**: GeoTIFF format, RGB or multispectral
- **Vector Annotations**: GeoJSON with tree crown polygons
- **Geospatial Metadata**: Rasterio-readable transforms (CRS, bounds)
- **Metadata CSV**: Coverage percentages per tile (for filtering)

## Output Artifacts

### Stage 2 Outputs
```
yolo_dataset/
├── images/
│   ├── train/     # Training images
│   └── val/       # Validation images
├── labels/
│   ├── train/     # YOLO annotations
│   └── val/       # YOLO annotations
└── dataset.yaml   # YOLO configuration
```

### Stage 3 Outputs
```
models/
├── model_1.pt                      # Trained weights
├── model_1_metrics.json            # Training metrics
├── model_1_coco_instances_results.json  # Evaluation results
└── ...
```

### Stage 4 Outputs
```
evaluation_results/
├── randcrowns_summary.csv          # RandCrowns metrics per tile
├── model_comparison.csv            # Multi-model comparison table
├── false_positives.gpkg            # False positive detections
├── predictions.geojson             # Model predictions
└── plots/                          # Visualization plots
```

## Dependencies

```
geopandas >= 0.12.0
rasterio >= 1.3.0
shapely >= 2.0.0
opencv-python >= 4.6.0
numpy >= 1.23.0
pandas >= 1.5.0
matplotlib >= 3.6.0
scikit-learn >= 1.2.0
ultralytics >= 8.0.0
torch >= 1.13.0
deepforest >= 1.0.0  (for model comparison)
```

## Installation

### Google Colab (Recommended)
The notebooks are optimized for Google Colab and handle installation automatically via `!pip install ...` cells.

### Local Environment
```bash
# Create conda environment
conda create -n tree-canopy python=3.10
conda activate tree-canopy

# Install dependencies
pip install geopandas rasterio shapely opencv-python ultralytics torch pandas matplotlib scikit-learn
```

## Usage

### Full Pipeline
1. Start with **00_create_vegetation_mask.ipynb**
2. Follow stages sequentially through Stage 4
3. Each notebook reads outputs from previous stage

### Single Model Training
- Use **02_02_Training_Data_Preparation.ipynb** to prepare YOLO dataset
- Choose training script: 03_01a, 03_01b, or 03_01c
- Run **04_01_Evaluation_RandCrowns.ipynb** for single model evaluation

### Model Comparison
- Train multiple models using different configs/loss functions
- Run **04_02_Evaluation_DeepForest_and_Dalponte.ipynb** for comparative analysis

## Important Notes

### Data Paths
All notebooks use Google Drive paths (`/content/drive/...`). Modify paths for local execution:
```python
# Colab
data_folder = "/content/drive/MyDrive/data/"

# Local
data_folder = "C:/path/to/data/"
```

### CRS Handling
Default CRS: **EPSG:25833** (UTM Zone 33N, Berlin region)
Modify in configuration sections for other regions.

### GPU Requirements
- Model training requires GPU (CUDA-enabled)
- Google Colab: Select GPU runtime (`Runtime > Change runtime type > GPU`)
- Local: Ensure PyTorch detects CUDA device

## References

- **YOLOv8**: Ultralytics instance segmentation https://github.com/ultralytics/ultralytics
- **DeepForest**: CNN for aerial tree detection https://github.com/weecology/DeepForest
- **RandCrowns**: Custom metric for tree crown evaluation (original research metric)

## License

[Specify your license here]

## Authors

[Your name/organization]

## Citation

```bibtex
[Add citation information if applicable]
```
