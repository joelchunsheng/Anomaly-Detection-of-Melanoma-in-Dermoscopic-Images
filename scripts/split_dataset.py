from src.data.split import create_data_splits

# To run script
# python -m scripts.split_dataset

# test, train, val csv added to data/splits

def main():
    create_data_splits(
        input_csv="data/raw/HAM10000/metadata.csv",
        output_dir="data/splits",
        train_size=0.7,
        val_size=0.15,
        test_size=0.15,
        random_state=42,
    )

if __name__ == "__main__":
    main()