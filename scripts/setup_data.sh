#!/usr/bin/env bash
set -e

echo "Starting dataset setup..."

DATA_DIR="data/raw"
DATASET_DIR="$DATA_DIR/HAM10000"

mkdir -p "$DATA_DIR"

if [ ! -d "$DATASET_DIR" ]; then
    echo "Cloning HAM10000 dataset into $DATASET_DIR ..."
    git clone https://huggingface.co/datasets/Nagabu/HAM10000 "$DATASET_DIR"
else
    echo "Dataset folder already exists at $DATASET_DIR, skipping clone."
fi

echo "Running extraction script..."
python scripts/extract_ham10000.py

echo "Running split script..."
python -m scripts.split_dataset

echo "Dataset setup complete."

# To run set up script 
# chmod +x scripts/setup_data.sh
# ./scripts/setup_data.sh