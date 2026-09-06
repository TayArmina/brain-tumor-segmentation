from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch

import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm

from .config import GlobalConfig, ExperimentConfig


# ---------------------------------------------------------
# Dataset scanning
# ---------------------------------------------------------

def build_dataframe(root: str) -> pd.DataFrame:
    """Find MRI images and their corresponding segmentation masks."""

    root = Path(root)
    records = []

    patient_dirs = sorted(
        path for path in root.iterdir()
        if path.is_dir()
    )

    for patient_dir in tqdm(
        patient_dirs,
        desc="Scanning patients",
    ):
        for image_path in sorted(patient_dir.glob("*.tif")):

            if image_path.stem.endswith("_mask"):
                continue

            mask_path = image_path.with_name(
                f"{image_path.stem}_mask.tif"
            )

            if mask_path.exists():
                records.append({
                    "patient": patient_dir.name,
                    "image": str(image_path),
                    "mask": str(mask_path),
                })

    dataframe = pd.DataFrame(records)

    if dataframe.empty:
        raise RuntimeError(
            f"No valid image-mask pairs found under {root}"
        )

    return dataframe


# ---------------------------------------------------------
# Mask information
# ---------------------------------------------------------

def read_mask_information(mask_path: str) -> Dict:
    """Calculate tumor statistics from one segmentation mask."""

    mask = cv2.imread(
        mask_path,
        cv2.IMREAD_GRAYSCALE,
    )

    if mask is None:
        raise FileNotFoundError(
            f"Could not read mask: {mask_path}"
        )

    height, width = mask.shape

    area = int(
        (mask > 0).sum()
    )

    ratio = float(
        area / (height * width)
    )

    return {
        "mask_area": area,
        "mask_ratio": ratio,
        "height": height,
        "width": width,
        "has_tumor": int(area > 0),
    }


def add_mask_information(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Add tumor statistics to the dataset table."""

    information = [
        read_mask_information(path)
        for path in tqdm(
            dataframe["mask"],
            desc="Reading masks",
        )
    ]

    information = pd.DataFrame(information)

    return pd.concat(
        [
            dataframe.reset_index(drop=True),
            information,
        ],
        axis=1,
    )


# ---------------------------------------------------------
# Tumor size groups
# ---------------------------------------------------------

def tumor_size_from_ratio(
    mask_ratio: float,
) -> str:

    if mask_ratio == 0:
        return "none"

    if mask_ratio < 0.005:
        return "small"

    if mask_ratio < 0.03:
        return "medium"

    return "large"


def add_tumor_size_groups(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    dataframe = dataframe.copy()

    dataframe["tumor_size"] = (
        dataframe["mask_ratio"]
        .apply(tumor_size_from_ratio)
    )

    return dataframe


# ---------------------------------------------------------
# Patient-level information
# ---------------------------------------------------------

def build_patient_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    patient_summary = (
        dataframe.groupby("patient")
        .agg(
            image_count=("image", "size"),
            tumor_image_count=("has_tumor", "sum"),
            tumor_ratio=("has_tumor", "mean"),
            total_tumor_area=("mask_area", "sum"),
        )
        .reset_index()
    )

    try:
        patient_summary["burden_group"] = pd.qcut(
            patient_summary["tumor_ratio"],
            q=4,
            labels=False,
            duplicates="drop",
        ).astype(str)

    except ValueError:
        patient_summary["burden_group"] = (
            patient_summary["tumor_ratio"] > 0
        ).astype(int).astype(str)

    return patient_summary


# ---------------------------------------------------------
# Patient-level train/validation/test split
# ---------------------------------------------------------

def patient_level_split(
    dataframe: pd.DataFrame,
    patient_table: pd.DataFrame,
    seed: int = 42,
    train_fraction: float = 0.80,
    validation_fraction: float = 0.10,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:

    patients = patient_table["patient"].values
    strata = patient_table["burden_group"].values

    try:
        train_patients, temporary_patients = train_test_split(
            patients,
            train_size=train_fraction,
            random_state=seed,
            stratify=strata,
        )

    except ValueError:
        train_patients, temporary_patients = train_test_split(
            patients,
            train_size=train_fraction,
            random_state=seed,
            shuffle=True,
        )

    temporary_table = patient_table[
        patient_table["patient"].isin(
            temporary_patients
        )
    ].reset_index(drop=True)

    relative_validation_fraction = (
        validation_fraction
        / (1 - train_fraction)
    )

    try:
        validation_patients, test_patients = train_test_split(
            temporary_table["patient"].values,
            train_size=relative_validation_fraction,
            random_state=seed,
            stratify=temporary_table[
                "burden_group"
            ].values,
        )

    except ValueError:
        validation_patients, test_patients = train_test_split(
            temporary_table["patient"].values,
            train_size=relative_validation_fraction,
            random_state=seed,
            shuffle=True,
        )

    train_df = dataframe[
        dataframe["patient"].isin(
            train_patients
        )
    ].reset_index(drop=True)

    validation_df = dataframe[
        dataframe["patient"].isin(
            validation_patients
        )
    ].reset_index(drop=True)

    test_df = dataframe[
        dataframe["patient"].isin(
            test_patients
        )
    ].reset_index(drop=True)

    return (
        train_df,
        validation_df,
        test_df,
    )


# ---------------------------------------------------------
# Leakage check
# ---------------------------------------------------------

def check_patient_leakage(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Ensure that the same patient never appears in two splits."""

    assert set(
        train_df["patient"]
    ).isdisjoint(
        validation_df["patient"]
    )

    assert set(
        train_df["patient"]
    ).isdisjoint(
        test_df["patient"]
    )

    assert set(
        validation_df["patient"]
    ).isdisjoint(
        test_df["patient"]
    )


# ---------------------------------------------------------
# Save/load fixed dataset splits
# ---------------------------------------------------------

def save_splits(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    results_dir: str,
) -> None:

    split_directory = (
        Path(results_dir)
        / "splits"
    )

    split_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_df.to_csv(
        split_directory / "train.csv",
        index=False,
    )

    validation_df.to_csv(
        split_directory / "validation.csv",
        index=False,
    )

    test_df.to_csv(
        split_directory / "test.csv",
        index=False,
    )


def load_splits(
    results_dir: str,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:

    split_directory = (
        Path(results_dir)
        / "splits"
    )

    train_path = (
        split_directory
        / "train.csv"
    )

    validation_path = (
        split_directory
        / "validation.csv"
    )

    test_path = (
        split_directory
        / "test.csv"
    )

    if not (
        train_path.exists()
        and validation_path.exists()
        and test_path.exists()
    ):
        raise FileNotFoundError(
            "Saved dataset splits were not found."
        )

    train_df = pd.read_csv(train_path)
    validation_df = pd.read_csv(validation_path)
    test_df = pd.read_csv(test_path)

    return (
        train_df,
        validation_df,
        test_df,
    )


# ---------------------------------------------------------
# Data augmentation
# ---------------------------------------------------------

def get_transforms(
    image_size: int,
    use_augmentation: bool,
) -> Tuple[
    A.Compose,
    A.Compose,
]:

    evaluation_transform = A.Compose([
        A.Resize(
            image_size,
            image_size,
        ),
        A.Normalize(),
        ToTensorV2(),
    ])

    if not use_augmentation:
        return (
            evaluation_transform,
            evaluation_transform,
        )

    training_transform = A.Compose([

        A.HorizontalFlip(p=0.5),

        A.VerticalFlip(p=0.1),

        A.OneOf([
            A.RandomRotate90(p=1.0),

            A.Affine(
                scale=(0.95, 1.05),
                translate_percent=(-0.03, 0.03),
                rotate=(-10, 10),
                shear=(-3, 3),
                p=1.0,
            ),
        ], p=0.5),

        A.ElasticTransform(
            alpha=10,
            sigma=5,
            p=0.1,
        ),

        A.OneOf([
            A.GaussNoise(
                std_range=(0.01, 0.05),
                p=1.0,
            ),

            A.GaussianBlur(
                blur_limit=(3, 5),
                p=1.0,
            ),

            A.MotionBlur(
                blur_limit=(3, 5),
                p=1.0,
            ),
        ], p=0.2),

        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=1.0,
            ),

            A.RandomGamma(
                gamma_limit=(90, 110),
                p=1.0,
            ),

            A.CLAHE(
                clip_limit=2.0,
                p=1.0,
            ),
        ], p=0.25),

        A.Resize(
            image_size,
            image_size,
        ),

        A.Normalize(),

        ToTensorV2(),
    ])

    return (
        training_transform,
        evaluation_transform,
    )


# ---------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------

class BrainTumorDataset(Dataset):

    def __init__(
        self,
        dataframe: pd.DataFrame,
        transform: Optional[A.Compose] = None,
        cache_images: bool = False,
    ):

        self.dataframe = (
            dataframe
            .reset_index(drop=True)
            .copy()
        )

        self.transform = transform
        self.cache_images = cache_images

        self.cache: Dict[
            int,
            Tuple[
                np.ndarray,
                np.ndarray,
            ],
        ] = {}

        if self.cache_images:
            self._build_cache()


    def _read_pair(
        self,
        index: int,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
    ]:

        row = self.dataframe.iloc[index]

        image = cv2.imread(
            row["image"],
            cv2.IMREAD_COLOR,
        )

        mask = cv2.imread(
            row["mask"],
            cv2.IMREAD_GRAYSCALE,
        )

        if image is None:
            raise FileNotFoundError(
                f"Could not read image: {row['image']}"
            )

        if mask is None:
            raise FileNotFoundError(
                f"Could not read mask: {row['mask']}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        mask = (
            mask > 0
        ).astype(np.uint8)

        return image, mask


    def _build_cache(self) -> None:

        for index in tqdm(
            range(len(self.dataframe)),
            desc="Caching images",
            leave=False,
        ):
            self.cache[index] = (
                self._read_pair(index)
            )


    def __len__(self) -> int:
        return len(self.dataframe)


    def __getitem__(
        self,
        index: int,
    ) -> Dict:

        if self.cache_images:

            image, mask = self.cache[index]

            image = image.copy()
            mask = mask.copy()

        else:
            image, mask = (
                self._read_pair(index)
            )

        if self.transform is not None:

            transformed = self.transform(
                image=image,
                mask=mask,
            )

            image = transformed["image"]
            mask = transformed["mask"]

        mask = (
            mask
            .float()
            .unsqueeze(0)
        )

        row = self.dataframe.iloc[index]

        return {
            "image": image.float(),

            "mask": mask,

            "has_tumor": torch.tensor(
                row["has_tumor"],
                dtype=torch.bool,
            ),

            "mask_ratio": torch.tensor(
                row["mask_ratio"],
                dtype=torch.float32,
            ),

            "patient": row["patient"],

            "image_path": row["image"],
        }


# ---------------------------------------------------------
# Weighted sampling
# ---------------------------------------------------------

SAMPLER_WEIGHTS = {
    "none": 1.0,
    "small": 3.0,
    "medium": 2.0,
    "large": 1.5,
}


def create_weighted_sampler(
    dataframe: pd.DataFrame,
) -> WeightedRandomSampler:

    sample_weights = (
        dataframe["tumor_size"]
        .map(SAMPLER_WEIGHTS)
        .astype(np.float64)
        .values
    )

    return WeightedRandomSampler(
        weights=torch.as_tensor(
            sample_weights,
            dtype=torch.double,
        ),
        num_samples=len(sample_weights),
        replacement=True,
    )


# ---------------------------------------------------------
# DataLoader creation
# ---------------------------------------------------------

def build_dataloaders(
    experiment: ExperimentConfig,
    config: GlobalConfig,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[
    DataLoader,
    DataLoader,
    DataLoader,
]:

    train_transform, evaluation_transform = get_transforms(
        image_size=config.image_size,
        use_augmentation=experiment.use_augmentation,
    )

    train_dataset = BrainTumorDataset(
        dataframe=train_df,
        transform=train_transform,
        cache_images=config.cache_images,
    )

    validation_dataset = BrainTumorDataset(
        dataframe=validation_df,
        transform=evaluation_transform,
        cache_images=config.cache_images,
    )

    test_dataset = BrainTumorDataset(
        dataframe=test_df,
        transform=evaluation_transform,
        cache_images=config.cache_images,
    )

    sampler = (
        create_weighted_sampler(
            train_df
        )
        if config.use_weighted_sampler
        else None
    )

    loader_arguments = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": (
            config.num_workers > 0
        ),
    }

    train_loader = DataLoader(
        train_dataset,
        sampler=sampler,
        shuffle=sampler is None,
        drop_last=False,
        **loader_arguments,
    )

    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        drop_last=False,
        **loader_arguments,
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        drop_last=False,
        **loader_arguments,
    )

    return (
        train_loader,
        validation_loader,
        test_loader,
    )