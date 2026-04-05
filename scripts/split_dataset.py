from src.data.split import create_data_splits

# To run script:
# python -m scripts.split_dataset

# Outputs train.csv, val.csv, test.csv to data_new/splits/

def main():
    create_data_splits(
        train_metadata_csv="data_new/raw/HAM10000_metadata",
        test_groundtruth_csv="data_new/raw/ISIC2018_Task3_Test_GroundTruth.csv",
        output_dir="data_new/splits",
        val_size=0.2,
        random_state=42,
    )

if __name__ == "__main__":
    main()
