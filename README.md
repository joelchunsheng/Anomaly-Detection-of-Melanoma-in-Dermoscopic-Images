# 50.039 Deep Learning Y2026 
Anomaly Detection of Melanoma in Dermoscopic Images

## Project Structure

```text
├── data_new/
│   ├── images/
│   │   ├── train/                        # Training + validation images (HAM10000)
│   │   └── test/                         # Test images (ISIC 2018 Task 3)
│   ├── raw/
│   │   ├── HAM10000_metadata/            # Patient metadata CSVs (age, sex, localization)
│   │   └── ISIC2018_Task3_Test_GroundTruth.csv
│   └── splits/
│       ├── train.csv                     # Training split labels
│       ├── val.csv                       # Validation split labels
│       └── test.csv                      # Test split labels
│
├── docs/
│   ├── model_report.md                   # Full results table and analysis for all architecture families
│   ├── recreate_best_models.md           # Step-by-step guide to retrain the best models from scratch
│   ├── evals_usage.md                    # Migration guide for the shared evaluation utilities
│   └── notebook_structure.md            # Conventions and structure used across experiment notebooks
│
├── models/                               # Saved model checkpoints (.pth)
│
├── notebooks/
│   ├── Group16.ipynb                     # Submission notebook: loads best checkpoints, runs ensemble TTA, evaluates
│   ├── cnn/                              # Custom CNN baseline experiments (01–05)
│   ├── densenet/                         # DenseNet transfer learning experiments (01–05)
│   ├── efficientnet/                     # EfficientNet-B0/B3/B4 experiments (01–10)
│   ├── mobilenet/                        # MobileNetV3-Large experiments + ensemble (01–02)
│   ├── resnet/                           # ResNet-18 experiments (01–06)
│   ├── resnet50/                         # ResNet-50 experiments (01–08)
│   ├── vit/                              # Vision Transformer (ViT-B/16) experiments (01–03)
│   ├── focal_loss_experiments/           # Cross-architecture focal loss ablations
│   └── penalty_experiments/             # Cross-architecture L1/L2 penalty ablations
│
├── scripts/
│   ├── setup_data.py                     # Downloads HAM10000 from Kaggle and runs the full pipeline
│   ├── split_dataset.py                  # Creates stratified train/val/test splits
│   ├── preprocess_images.py              # Resizes and saves images at 224×224
│   └── preprocess_images_380.py          # Resizes and saves images at 380×380 (for EfficientNet-B4)
│
├── src/
│   ├── data/
│   │   ├── dataset.py                    # Dataset classes: HAM10000Dataset and HAM10000DatasetWithMetadata
│   │   ├── dataloader.py                 # get_dataloaders() factory for standard (image-only) pipelines
│   │   ├── transform.py                  # Train augmentation and eval transforms
│   │   └── split.py                      # Stratified split logic
│   ├── models/
│   │   ├── efficientnet.py               # EfficientNetB0WithMetadata, EfficientNetB4WithMetadata
│   │   ├── mobilenet.py                  # MobileNetV3LargeWithMetadata
│   │   ├── resnet.py                     # ResNet-18/50 with optional metadata fusion
│   │   ├── vit.py                        # ViT-B/16 wrapper
│   │   ├── cnn_baseline.py               # SimpleCNN (3-conv baseline)
│   │   ├── cnn_batchnorm.py              # BatchNormCNN
│   │   ├── cnn_batchnorm_deeper.py       # DeeperBatchNormCNN
│   │   └── cnn_batchnorm_residual.py     # ResidualBatchNormCNN
│   ├── training/
│   │   ├── trainer.py                    # train_one_epoch() and validate_one_epoch() — support metadata and L1/L2
│   │   └── losses.py                     # Focal loss implementation
│   └── utils/
│       ├── __init__.py                   # Exports: seed_everything, plot_training_curves, find_best_threshold, evaluate_model
│       ├── evaluation.py                 # evaluate_model(), find_best_threshold(), plot_training_curves()
│       └── seed.py                       # seed_everything() and seed_worker() for reproducibility
│
├── streamlit/
│   ├── app.py                            # Streamlit app entry point
│   └── app_src/
│       ├── config.py                     # App configuration (model paths, class names, thresholds)
│       ├── model_utils.py                # Loads checkpoints and runs inference
│       └── ui_components.py             # Reusable UI widgets
│
├── requirements.txt                      # Python dependencies
└── README.md                             # This file
```

## Setup Instructions

### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

> On macOS, use `source .venv/bin/activate` to activate the virtual environment. <br/>
> On Windows PowerShell, use `.\.venv\Scripts\Activate.ps1`.

### 2. Install requirements

```bash
pip install -r requirements.txt
```

### 3. Download and prepare the dataset

This project uses the HAM10000 / ISIC 2018 dataset. There are two ways to get it:

**Option A — Kaggle CLI (automated)**

Make sure you have the Kaggle CLI configured:

```bash
pip install kaggle
mkdir -p ~/.kaggle
cp /path/to/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

Then run:

```bash
python scripts/setup_data.py
```

This script will download the dataset, generate train/validation/test splits, and preprocess images automatically.

**Option B — Manual download**

Download the dataset directly from [https://www.kaggle.com/datasets/nightfury007/ham10000-isic2018-raw](https://www.kaggle.com/datasets/nightfury007/ham10000-isic2018-raw) and extract it into `data_new/raw/`. Then run the split and preprocess steps manually:

```bash
python -m scripts.split_dataset
python -m scripts.preprocess_images
```

### 4. Recreate the best models

To train the two model checkpoints from scratch and reproduce the ensemble result (AUC 0.9235), follow the step-by-step guide in [`docs/recreate_best_models.md`](docs/recreate_best_models.md).

For results, analysis, and iteration reasoning across all architecture families (CNN, DenseNet, ResNet-18, ResNet-50, ViT, EfficientNet, MobileNet), see [`docs/model_report.md`](docs/model_report.md), or our report for further details.

### 5. Run the Streamlit app

After environment setup and dataset preparation, access our demo via the Streamlit app with:

```bash
streamlit run streamlit/app.py
```

Then open the URL shown in the terminal, usually `http://localhost:8501`.

