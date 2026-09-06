import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Set random seeds for reproducible experiments."""

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class GlobalConfig:
    # Dataset
    data_root: str = "data/kaggle_3m"

    # Image/model settings
    image_size: int = 256
    in_channels: int = 3
    num_classes: int = 1
    base_channels: int = 32

    # Training
    batch_size: int = 8
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5

    num_workers: int = 2
    cache_images: bool = True
    use_weighted_sampler: bool = True

    use_amp: bool = True
    gradient_clip: float = 1.0

    early_stopping_patience: int = 10
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5

    threshold: float = 0.5
    seed: int = 42

    # Output folder
    results_dir: str = "results"

    # Hardware
    device: str = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


@dataclass
class ExperimentConfig:
    name: str

    use_residual: bool = True
    use_cbam: bool = True
    use_augmentation: bool = True

    use_aspp: bool = True
    use_deep_supervision: bool = True

    # Options:
    # "dice_focal"
    # "dice_focal_tversky"
    loss_name: str = "dice_focal"


cfg = GlobalConfig()

Path(cfg.results_dir).mkdir(
    parents=True,
    exist_ok=True,
)

seed_everything(cfg.seed)