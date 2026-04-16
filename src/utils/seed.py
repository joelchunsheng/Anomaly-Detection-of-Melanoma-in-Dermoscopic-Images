import os
import random

import numpy as np
import torch


def seed_worker(worker_id):
    """Seed numpy and random in each DataLoader worker.

    Pass as worker_init_fn to any DataLoader that uses num_workers > 0.
    The worker's torch seed is already set by the generator passed to the
    DataLoader; this function extends that to numpy and random so that
    augmentation transforms are also deterministic.
    """
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def seed_everything(seed: int = 42) -> torch.Generator:
    """Fully deterministic setup for training.

    Sets seeds for random, numpy, torch, and CUDA, enables cuDNN
    determinism, and returns a seeded Generator to pass to DataLoader.

    Usage::

        from src.utils import seed_everything, seed_worker

        g = seed_everything(42)

        train_loader = DataLoader(
            dataset,
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=g,
        )

    Returns:
        g: torch.Generator seeded with ``seed`` — pass as ``generator``
           to every DataLoader that uses shuffle=True.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    g = torch.Generator()
    g.manual_seed(seed)
    return g
