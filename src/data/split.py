from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def create_data_splits(
    input_csv,
    output_dir,
    train_size=0.7,
    val_size=0.15,
    test_size=0.15,
    random_state=42,
):

    input_csv = Path(input_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)

    train_df, temp_df = train_test_split(
        df,
        test_size=(1 - train_size),
        stratify=df["label"],
        random_state=random_state,
    )

    val_ratio_adjusted = val_size / (val_size + test_size)

    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_ratio_adjusted),
        stratify=temp_df["label"],
        random_state=random_state,
    )

    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    print("Data split completed.")
    print(f"Train set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    print(f"Test set: {len(test_df)} samples")