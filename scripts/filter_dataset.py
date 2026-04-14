"""
One-time preprocessing script to filter out invalid dataset entries.

Reads from:
  data_new/raw/HAM10000_metadata.csv        (original metadata CSV)
  data_new/raw/images/                     (image directory)

Writes to:
  data_new/raw/HAM10000_metadata_clean.csv (filtered metadata CSV)

Removes:
  - Rows where the corresponding image file does not exist

Run with:
  python -m scripts.filter_dataset
"""

import os
import pandas as pd

def main():
    csv_path = "data_new/raw/HAM10000_metadata.csv"
    image_dir = "data_new/raw/images"
    output_path = "data_new/raw/HAM10000_metadata_clean.csv"

    df = pd.read_csv(csv_path)

    def image_exists(img_id):
        return any(
            os.path.exists(os.path.join(image_dir, f"{img_id}.{ext}"))
            for ext in ["jpg", "jpeg", "png"]
        )

    df_filtered = df[df['image_id'].apply(image_exists)]

    print(f"Original: {len(df)}")
    print(f"Filtered: {len(df_filtered)}")
    print(f"Removed: {len(df) - len(df_filtered)}")

    df_filtered.to_csv(output_path, index=False)

if __name__ == "__main__":
    main()