import gc
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm.auto import tqdm

from .config import GlobalConfig, ExperimentConfig, seed_everything
from .data import build_dataloaders
from .model import build_model


# ---------------------------------------------------------
# Segmentation losses
# ---------------------------------------------------------

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        probabilities = torch.sigmoid(logits)

        probabilities = probabilities.flatten(start_dim=1)
        targets = targets.flatten(start_dim=1)

        intersection = (
            probabilities * targets
        ).sum(dim=1)

        denominator = (
            probabilities.sum(dim=1)
            + targets.sum(dim=1)
        )

        dice = (
            2 * intersection + self.smooth
        ) / (
            denominator + self.smooth
        )

        return 1 - dice.mean()


class BinaryFocalLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.8,
        gamma: float = 2.0,
    ):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )

        probability_correct = torch.exp(-bce)

        focal_loss = (
            self.alpha
            * (1 - probability_correct).pow(self.gamma)
            * bce
        )

        return focal_loss.mean()


class TverskyLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        smooth: float = 1.0,
    ):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        probabilities = torch.sigmoid(logits)

        probabilities = probabilities.flatten(start_dim=1)
        targets = targets.flatten(start_dim=1)

        true_positive = (
            probabilities * targets
        ).sum(dim=1)

        false_positive = (
            probabilities * (1 - targets)
        ).sum(dim=1)

        false_negative = (
            (1 - probabilities) * targets
        ).sum(dim=1)

        score = (
            true_positive + self.smooth
        ) / (
            true_positive
            + self.alpha * false_positive
            + self.beta * false_negative
            + self.smooth
        )

        return 1 - score.mean()


# ---------------------------------------------------------
# Combined losses
# ---------------------------------------------------------

class DiceFocalSegmentationLoss(nn.Module):
    def __init__(
        self,
        dice_weight: float = 0.50,
        focal_weight: float = 0.50,
    ):
        super().__init__()

        if not np.isclose(
            dice_weight + focal_weight,
            1.0,
        ):
            raise ValueError(
                "Dice and Focal weights must sum to 1."
            )

        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

        self.dice_loss = DiceLoss()

        self.focal_loss = BinaryFocalLoss(
            alpha=0.8,
            gamma=2.0,
        )

    def single_output_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        return (
            self.dice_weight
            * self.dice_loss(logits, targets)
            +
            self.focal_weight
            * self.focal_loss(logits, targets)
        )

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> torch.Tensor:

        main_loss = self.single_output_loss(
            outputs["main"],
            targets,
        )

        if (
            "deep_2" in outputs
            and "deep_3" in outputs
        ):
            return (
                0.70 * main_loss
                + 0.20 * self.single_output_loss(
                    outputs["deep_2"],
                    targets,
                )
                + 0.10 * self.single_output_loss(
                    outputs["deep_3"],
                    targets,
                )
            )

        return main_loss


class DiceFocalTverskySegmentationLoss(nn.Module):
    def __init__(
        self,
        dice_weight: float = 0.25,
        focal_weight: float = 0.15,
        tversky_weight: float = 0.60,
    ):
        super().__init__()

        total = (
            dice_weight
            + focal_weight
            + tversky_weight
        )

        if not np.isclose(total, 1.0):
            raise ValueError(
                "Loss weights must sum to 1."
            )

        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight

        self.dice_loss = DiceLoss()

        self.focal_loss = BinaryFocalLoss(
            alpha=0.8,
            gamma=2.0,
        )

        # beta > alpha gives more penalty
        # to false-negative tumor pixels.
        self.tversky_loss = TverskyLoss(
            alpha=0.30,
            beta=0.70,
        )

    def single_output_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        return (
            self.dice_weight
            * self.dice_loss(logits, targets)
            +
            self.focal_weight
            * self.focal_loss(logits, targets)
            +
            self.tversky_weight
            * self.tversky_loss(logits, targets)
        )

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> torch.Tensor:

        main_loss = self.single_output_loss(
            outputs["main"],
            targets,
        )

        if (
            "deep_2" in outputs
            and "deep_3" in outputs
        ):
            return (
                0.70 * main_loss
                + 0.20 * self.single_output_loss(
                    outputs["deep_2"],
                    targets,
                )
                + 0.10 * self.single_output_loss(
                    outputs["deep_3"],
                    targets,
                )
            )

        return main_loss


def build_criterion(
    experiment: ExperimentConfig,
) -> nn.Module:

    if experiment.loss_name == "dice_focal":
        return DiceFocalSegmentationLoss()

    if experiment.loss_name == "dice_focal_tversky":
        return DiceFocalTverskySegmentationLoss()

    raise ValueError(
        f"Unsupported loss: {experiment.loss_name}"
    )


# ---------------------------------------------------------
# Segmentation metrics
# ---------------------------------------------------------

@torch.no_grad()
def batch_metric_tensors(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    epsilon: float = 1e-7,
) -> Dict[str, torch.Tensor]:

    predictions = (
        torch.sigmoid(logits)
        >= threshold
    ).float()

    predictions = predictions.flatten(start_dim=1)
    targets = targets.flatten(start_dim=1)

    true_positive = (
        predictions * targets
    ).sum(dim=1)

    false_positive = (
        predictions * (1 - targets)
    ).sum(dim=1)

    false_negative = (
        (1 - predictions) * targets
    ).sum(dim=1)

    true_negative = (
        (1 - predictions)
        * (1 - targets)
    ).sum(dim=1)

    target_positive = (
        targets.sum(dim=1) > 0
    )

    prediction_positive = (
        predictions.sum(dim=1) > 0
    )

    dice = (
        2 * true_positive + epsilon
    ) / (
        2 * true_positive
        + false_positive
        + false_negative
        + epsilon
    )

    iou = (
        true_positive + epsilon
    ) / (
        true_positive
        + false_positive
        + false_negative
        + epsilon
    )

    precision = (
        true_positive + epsilon
    ) / (
        true_positive
        + false_positive
        + epsilon
    )

    recall = (
        true_positive + epsilon
    ) / (
        true_positive
        + false_negative
        + epsilon
    )

    specificity = (
        true_negative + epsilon
    ) / (
        true_negative
        + false_positive
        + epsilon
    )

    false_positive_empty = (
        prediction_positive
        & (~target_positive)
    ).float()

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "target_positive": target_positive,
        "false_positive_empty": false_positive_empty,
    }


class SegmentationMetricAccumulator:
    def __init__(
        self,
        threshold: float = 0.5,
    ):
        self.threshold = threshold

        self.values: Dict[
            str,
            List[torch.Tensor],
        ] = {
            "dice": [],
            "iou": [],
            "precision": [],
            "recall": [],
            "specificity": [],
            "target_positive": [],
            "false_positive_empty": [],
        }

    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:

        values = batch_metric_tensors(
            logits,
            targets,
            threshold=self.threshold,
        )

        for key, value in values.items():
            self.values[key].append(
                value.detach().cpu()
            )

    def compute(
        self,
    ) -> Dict[str, float]:

        values = {
            key: torch.cat(value)
            for key, value
            in self.values.items()
        }

        positive = (
            values["target_positive"].bool()
        )

        empty = ~positive

        def safe_mean(
            tensor: torch.Tensor,
        ) -> float:

            if tensor.numel() == 0:
                return float("nan")

            return float(
                tensor.float().mean().item()
            )

        return {
            "dice_all":
                safe_mean(values["dice"]),

            "dice_tumor":
                safe_mean(
                    values["dice"][positive]
                ),

            "dice_empty":
                safe_mean(
                    values["dice"][empty]
                ),

            "iou_all":
                safe_mean(values["iou"]),

            "iou_tumor":
                safe_mean(
                    values["iou"][positive]
                ),

            "precision_all":
                safe_mean(
                    values["precision"]
                ),

            "recall_tumor":
                safe_mean(
                    values["recall"][positive]
                ),

            "specificity_all":
                safe_mean(
                    values["specificity"]
                ),

            "empty_false_positive_rate":
                safe_mean(
                    values[
                        "false_positive_empty"
                    ][empty]
                ),

            "num_images":
                int(values["dice"].numel()),

            "num_tumor_images":
                int(positive.sum().item()),

            "num_empty_images":
                int(empty.sum().item()),
        }


# ---------------------------------------------------------
# Average loss meter
# ---------------------------------------------------------

class AverageMeter:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(
        self,
        value: float,
        number: int = 1,
    ) -> None:

        self.total += value * number
        self.count += number

    @property
    def average(self) -> float:

        return (
            self.total
            / max(self.count, 1)
        )


# ---------------------------------------------------------
# One training epoch
# ---------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler,
    criterion: nn.Module,
    config: GlobalConfig,
) -> Dict[str, float]:

    model.train()

    loss_meter = AverageMeter()

    metrics = SegmentationMetricAccumulator(
        threshold=config.threshold,
    )

    progress = tqdm(
        loader,
        desc="Train",
        leave=False,
    )

    amp_enabled = (
        config.use_amp
        and torch.cuda.is_available()
    )

    for batch in progress:

        images = batch["image"].to(
            config.device,
            non_blocking=True,
        )

        masks = batch["mask"].to(
            config.device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.autocast(
            device_type="cuda",
            enabled=amp_enabled,
        ):
            outputs = model(images)
            loss = criterion(
                outputs,
                masks,
            )

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=config.gradient_clip,
        )

        scaler.step(optimizer)
        scaler.update()

        loss_meter.update(
            loss.item(),
            images.size(0),
        )

        metrics.update(
            outputs["main"],
            masks,
        )

        progress.set_postfix(
            loss=f"{loss_meter.average:.4f}"
        )

    result = metrics.compute()
    result["loss"] = loss_meter.average

    return result


# ---------------------------------------------------------
# Validation / test epoch
# ---------------------------------------------------------

@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    config: GlobalConfig,
    criterion: Optional[nn.Module] = None,
    description: str = "Evaluate",
) -> Dict[str, float]:

    model.eval()

    loss_meter = AverageMeter()

    metrics = SegmentationMetricAccumulator(
        threshold=config.threshold,
    )

    for batch in tqdm(
        loader,
        desc=description,
        leave=False,
    ):

        images = batch["image"].to(
            config.device,
            non_blocking=True,
        )

        masks = batch["mask"].to(
            config.device,
            non_blocking=True,
        )

        outputs = model(images)

        if criterion is not None:

            loss = criterion(
                outputs,
                masks,
            )

            loss_meter.update(
                loss.item(),
                images.size(0),
            )

        metrics.update(
            outputs["main"],
            masks,
        )

    result = metrics.compute()

    if criterion is not None:
        result["loss"] = loss_meter.average

    return result


# ---------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------

def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    validation_metrics: Dict[str, float],
    experiment: ExperimentConfig,
    config: GlobalConfig,
) -> None:

    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state":
                optimizer.state_dict(),

            "scheduler_state":
                scheduler.state_dict(),

            "scaler_state":
                scaler.state_dict(),

            "validation_metrics":
                validation_metrics,

            "experiment":
                asdict(experiment),

            "global_config":
                asdict(config),
        },
        path,
    )


def load_model_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    device: str,
) -> Dict:

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state"]
    )

    return checkpoint


# ---------------------------------------------------------
# Train complete experiment
# ---------------------------------------------------------

def train_experiment(
    experiment: ExperimentConfig,
    config: GlobalConfig,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Dict:

    seed_everything(
        config.seed
    )

    experiment_directory = (
        Path(config.results_dir)
        / experiment.name
    )

    experiment_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        experiment_directory
        / "experiment_config.json",
        "w",
    ) as file:

        json.dump(
            {
                "experiment":
                    asdict(experiment),

                "global":
                    asdict(config),
            },
            file,
            indent=4,
        )

    (
        train_loader,
        validation_loader,
        test_loader,
    ) = build_dataloaders(
        experiment=experiment,
        config=config,
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df,
    )

    model = build_model(
        experiment,
        config,
    ).to(
        config.device
    )

    criterion = build_criterion(
        experiment
    ).to(
        config.device
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    scheduler = (
        torch.optim.lr_scheduler
        .ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
        )
    )

    amp_enabled = (
        config.use_amp
        and torch.cuda.is_available()
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
    )

    checkpoint_path = (
        experiment_directory
        / "best_model.pt"
    )

    best_validation_dice = -np.inf
    best_epoch = 0
    epochs_without_improvement = 0

    history = []

    start_time = time.time()

    for epoch in range(
        1,
        config.epochs + 1,
    ):

        print(
            f"\n[{experiment.name}] "
            f"Epoch {epoch}/{config.epochs}"
        )

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            criterion=criterion,
            config=config,
        )

        validation_metrics = evaluate_epoch(
            model=model,
            loader=validation_loader,
            config=config,
            criterion=criterion,
            description="Validation",
        )

        monitor_score = (
            validation_metrics[
                "dice_tumor"
            ]
        )

        if np.isnan(monitor_score):
            monitor_score = (
                validation_metrics[
                    "dice_all"
                ]
            )

        scheduler.step(
            monitor_score
        )

        epoch_record = {
            "epoch": epoch,

            "learning_rate":
                optimizer.param_groups[0]["lr"],

            **{
                f"train_{key}": value
                for key, value
                in train_metrics.items()
            },

            **{
                f"validation_{key}": value
                for key, value
                in validation_metrics.items()
            },
        }

        history.append(
            epoch_record
        )

        pd.DataFrame(
            history
        ).to_csv(
            experiment_directory
            / "history.csv",
            index=False,
        )

        print(
            f"Train loss="
            f"{train_metrics['loss']:.4f} | "
            f"Val loss="
            f"{validation_metrics['loss']:.4f} | "
            f"Val Dice tumor="
            f"{validation_metrics['dice_tumor']:.4f}"
        )

        if (
            monitor_score
            > best_validation_dice
        ):

            best_validation_dice = (
                monitor_score
            )

            best_epoch = epoch

            epochs_without_improvement = 0

            save_checkpoint(
                path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                validation_metrics=
                    validation_metrics,
                experiment=experiment,
                config=config,
            )

        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= config.early_stopping_patience
        ):
            print("Early stopping.")
            break

    training_minutes = (
        time.time() - start_time
    ) / 60

    load_model_checkpoint(
        model,
        checkpoint_path,
        config.device,
    )

    test_metrics = evaluate_epoch(
        model=model,
        loader=test_loader,
        config=config,
        criterion=criterion,
        description="Test",
    )

    parameter_count = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    result = {
        "experiment":
            experiment.name,

        "loss_name":
            experiment.loss_name,

        "residual":
            experiment.use_residual,

        "cbam":
            experiment.use_cbam,

        "augmentation":
            experiment.use_augmentation,

        "aspp":
            experiment.use_aspp,

        "deep_supervision":
            experiment.use_deep_supervision,

        "parameters":
            parameter_count,

        "best_epoch":
            best_epoch,

        "training_minutes":
            training_minutes,

        "best_validation_dice_tumor":
            best_validation_dice,

        **{
            f"test_{key}": value
            for key, value
            in test_metrics.items()
        },
    }

    with open(
        experiment_directory
        / "result.json",
        "w",
    ) as file:

        json.dump(
            result,
            file,
            indent=4,
        )

    del model
    del optimizer
    del scheduler
    del scaler

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result