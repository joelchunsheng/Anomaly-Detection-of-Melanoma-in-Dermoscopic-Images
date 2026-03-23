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

    data_dir = "data/raw"
    dataset_dir = os.path.join(data_dir, "HAM10000")

    os.makedirs(data_dir, exist_ok=True)

    # Step 1: Clone dataset
    if not os.path.exists(dataset_dir):
        print(f"Cloning dataset into {dataset_dir}...")
        run_command(f"git clone https://huggingface.co/datasets/Nagabu/HAM10000 {dataset_dir}")
    else:
        print("Dataset already exists, skipping clone.")

    # Step 2: Extract dataset
    print("Running extraction script...")
    run_command("python scripts/extract_ham10000.py")

    # Step 3: Split dataset
    print("Running split script...")
    run_command("python -m scripts.split_dataset")

    print("Dataset setup complete.")


if __name__ == "__main__":
    main()