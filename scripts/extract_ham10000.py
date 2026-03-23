import pandas as pd

#  Check parquet file

# df = pd.read_parquet("data/raw/HAM10000/data/train-00000-of-00001.parquet")
# print(df.columns)
# print(df.head())

import os
import io
import pandas as pd
from PIL import Image

PARQUET_PATH = "data/raw/HAM10000/data/train-00000-of-00001.parquet"
OUTPUT_DIR = "data/raw/HAM10000"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
METADATA_CSV = os.path.join(OUTPUT_DIR, "metadata.csv")

os.makedirs(IMAGES_DIR, exist_ok=True)

df = pd.read_parquet(PARQUET_PATH)

metadata_rows = []

for i, row in df.iterrows():
    image_dict = row["image"]
    label = row["label"]

    image_bytes = image_dict["bytes"]
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    filename = f"img_{i}.jpg"
    image_path = os.path.join(IMAGES_DIR, filename)
    image.save(image_path)

    metadata_rows.append({
        "image_id": filename,
        "label": label
    })

metadata_df = pd.DataFrame(metadata_rows)
metadata_df.to_csv(METADATA_CSV, index=False)

print("Done.")
print(f"Saved {len(metadata_df)} images to: {IMAGES_DIR}")
print(f"Saved metadata to: {METADATA_CSV}")
print(metadata_df.head())