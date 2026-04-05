from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class HAM10000Dataset(Dataset):

    def __init__(self, csv_path, image_dir, transform=None):
        self.csv_path = Path(csv_path)
        self.image_dir = Path(image_dir)
        self.transform = transform

        self.data = pd.read_csv(self.csv_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        image_name = row["image_id"] + ".jpg"
        label = int(row["label"])

        image_path = self.image_dir / image_name
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label