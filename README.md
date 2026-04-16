# 50.039 Deep Learning Y2026 
Anomaly Detection of Melanoma in Dermoscopic Images

## Project Structure

```text
melanoma-anomaly-detection/
│
├── data/                  # Raw, processed, and split dataset files
├── data_new/              # Alternative dataset storage and processed data
├── notebooks/             # Data exploration, experiments, and result visualization
├── src/                   # Main source code
│   ├── data/              # Dataset loading, preprocessing, transforms, and splits
│   ├── models/            # Model architectures
│   ├── training/          # Training loops, losses, and metrics
│   ├── evaluation/        # Testing, evaluation, and plots
│   └── utils/             # Utility functions
│
├── scripts/               # Command-line scripts for training and evaluation
├── checkpoints/           # Saved model weights
├── outputs/               # Figures, logs, and reports
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
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

This project uses the Kaggle dataset downloader in `scripts/setup_data.py`.

Before running it, make sure you have the Kaggle CLI configured:

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

This script will:
- download the HAM10000 / ISIC dataset from Kaggle
- generate train/validation/test splits
- preprocess images

### 4. Run the Streamlit app

After environment setup and dataset preparation, launch the Streamlit app with:

```bash
streamlit run streamlit/app.py
```

Then open the URL shown in the terminal, usually `http://localhost:8501`.

