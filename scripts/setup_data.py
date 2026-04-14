import os
import subprocess
import sys


def run_command(cmd):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Error running command: {cmd}")
        sys.exit(1)


def main():
    print("Starting dataset setup...")

    raw_dir = "data_new/raw"
    os.makedirs(raw_dir, exist_ok=True)

    # Step 1: Download dataset via Kaggle API
    # Requires kaggle.json at ~/.kaggle/kaggle.json
    # Install with: pip install kaggle
    dataset_marker = os.path.join(raw_dir, "HAM10000_metadata")
    if not os.path.exists(dataset_marker):
        print("Downloading dataset from Kaggle...")
        run_command(
            f"kaggle datasets download -d nightfury007/ham10000-isic2018-raw "
            f"-p {raw_dir} --unzip"
        )
    else:
        print("Dataset already exists, skipping download.")

    # Step 2: Filter out missing images
    print("Filtering dataset...")
    run_command("python -m scripts.filter_dataset") 

    # Step 3: Generate train/val/test splits
    print("Running split script...")
    run_command("python -m scripts.split_dataset")

    # Step 4: Preprocess images (resize to 224x224)
    print("Running image preprocessing script...")
    run_command("python -m scripts.preprocess_images")

    print("Dataset setup complete.")
    print("Splits written to data_new/splits/")


if __name__ == "__main__":
    main()
