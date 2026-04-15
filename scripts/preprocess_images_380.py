"""
One-time preprocessing script to resize all images to 380x380.

Reads from:
  data_new/raw/HAM10000_images_combined_600x450/  (train images)
  data_new/raw/ISIC2018_Task3_Test_Images/         (test images)

Writes to:
  data_new/images_380/train/
  data_new/images_380/test/

Run with:
  python -m scripts.preprocess_images_380
"""

from pathlib import Path
from PIL import Image
from tqdm import tqdm

SIZE = (380, 380)

SOURCES = [
    ("data_new/raw/HAM10000_images_combined_600x450", "data_new/images_380/train"),
    ("data_new/raw/ISIC2018_Task3_Test_Images",        "data_new/images_380/test"),
]


def resize_folder(src: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    images = list(src.glob("*.jpg"))
    print(f"Processing {len(images)} images: {src} -> {dst}")
    for img_path in tqdm(images):
        img = Image.open(img_path).convert("RGB")
        img = img.resize(SIZE, Image.LANCZOS)
        img.save(dst / img_path.name, format="JPEG", quality=95)


def main():
    for src_str, dst_str in SOURCES:
        resize_folder(Path(src_str), Path(dst_str))
    print("Done. All images resized to 380x380.")


if __name__ == "__main__":
    main()
