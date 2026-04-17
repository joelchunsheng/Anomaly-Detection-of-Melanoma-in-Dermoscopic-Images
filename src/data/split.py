from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def _map_label(dx_series):
    return (dx_series == "mel").astype(int)


def create_data_splits(
    train_metadata_csv,
    test_groundtruth_csv,
    output_dir,
    val_size=0.2,
    random_state=42,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Train / Val split ---
    df = pd.read_csv(train_metadata_csv)
    df["label"] = _map_label(df["dx"])

    gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    train_idx, val_idx = next(gss.split(df, groups=df["lesion_id"]))

    train_df = df.iloc[train_idx][["image_id", "label"]].reset_index(drop=True)
    val_df = df.iloc[val_idx][["image_id", "label"]].reset_index(drop=True)

    # --- Test split ---
    test_df = pd.read_csv(test_groundtruth_csv)
    test_df["label"] = _map_label(test_df["dx"])
    test_df = test_df[["image_id", "label"]].reset_index(drop=True)

    # ISIC_0035068 has no corresponding image file
    missing_images = {"ISIC_0035068"}
    train_df = train_df[~train_df["image_id"].isin(missing_images)].reset_index(drop=True)
    val_df = val_df[~val_df["image_id"].isin(missing_images)].reset_index(drop=True)
    test_df = test_df[~test_df["image_id"].isin(missing_images)].reset_index(drop=True)

    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    print("Data split completed.")
    print(f"Train set:      {len(train_df)} samples  ({train_df['label'].sum()} melanoma)")
    print(f"Validation set: {len(val_df)} samples  ({val_df['label'].sum()} melanoma)")
    print(f"Test set:       {len(test_df)} samples  ({test_df['label'].sum()} melanoma)")

    # Leakage check
    train_lesions = set(df.iloc[train_idx]["lesion_id"])
    val_lesions = set(df.iloc[val_idx]["lesion_id"])
    overlap = train_lesions & val_lesions
    assert len(overlap) == 0, f"Lesion overlap detected: {overlap}"
    print("Leakage check passed: no lesion_id overlap between train and val.")
