from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.split import create_data_splits

# To run script:
# python -m scripts.split_dataset

# Outputs train.csv, val.csv, test.csv to data_new/splits/

def main():
    raw_dir = PROJECT_ROOT / "data_new" / "raw" / "dataverse_files"
    output_dir = PROJECT_ROOT / "data_new" / "splits"

    create_data_splits(
        train_metadata_csv=PROJECT_ROOT / "data_new" / "raw" / "HAM10000_metadata_clean.csv",
        test_groundtruth_csv=raw_dir / "ISIC2018_Task3_Test_GroundTruth.csv",
        output_dir=output_dir,
        val_size=0.2,
        random_state=42,
    )

if __name__ == "__main__":
    main()
