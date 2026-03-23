# 50.039 Deep Learning Y2026 
Anomaly Detection of Melanoma in Dermoscopic Images

## Project Structure

```text
melanoma-anomaly-detection/
│
├── data/                  # Raw, processed, and split dataset files
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

## Dataset Setup

Run the following command:

```bash
python scripts/setup_data.py