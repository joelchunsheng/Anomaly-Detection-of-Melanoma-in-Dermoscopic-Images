from torch.utils.data import DataLoader

from src.data.dataset import HAM10000Dataset
from src.data.transform import get_train_transforms, get_eval_transforms


def get_dataloaders(
    train_csv: str,
    val_csv: str,
    test_csv: str,
    image_dir: str,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 0,
    transform_train=None,
):
    if transform_train is None:
        transform_train = get_train_transforms(image_size=image_size)

    train_dataset = HAM10000Dataset(
        csv_path=train_csv,
        image_dir=image_dir,
        transform=transform_train,
    )

    val_dataset = HAM10000Dataset(
        csv_path=val_csv,
        image_dir=image_dir,
        transform=get_eval_transforms(image_size=image_size),
    )

    test_dataset = HAM10000Dataset(
        csv_path=test_csv,
        image_dir=image_dir,
        transform=get_eval_transforms(image_size=image_size),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader