from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .config import GlobalConfig, ExperimentConfig
from .training import train_experiment


# ---------------------------------------------------------
# Representative predictions
# ---------------------------------------------------------

@torch.no_grad()
def show_representative_predictions(
    model: nn.Module,
    loader: DataLoader,
    config: GlobalConfig,
    threshold: float = 0.5,
    number_of_empty_samples: int = 3,
) -> pd.DataFrame:
    """
    Show worst, typical and best tumor examples,
    together with several tumor-free examples.
    """

    model.eval()

    tumor_samples = []
    empty_samples = []

    for batch_index, batch in enumerate(loader):

        images = batch["image"].to(
            config.device
        )

        masks = batch["mask"].to(
            config.device
        )

        outputs = model(images)

        if isinstance(outputs, dict):
            logits = outputs["main"]
        else:
            logits = outputs

        probabilities = torch.sigmoid(
            logits
        )

        predictions = (
            probabilities >= threshold
        ).float()

        for sample_index in range(
            images.size(0)
        ):

            image = (
                images[sample_index]
                .detach()
                .cpu()
            )

            mask = (
                masks[sample_index, 0]
                .detach()
                .cpu()
            )

            probability = (
                probabilities[
                    sample_index,
                    0,
                ]
                .detach()
                .cpu()
            )

            prediction = (
                predictions[
                    sample_index,
                    0,
                ]
                .detach()
                .cpu()
            )

            ground_truth_pixels = int(
                mask.sum().item()
            )

            predicted_pixels = int(
                prediction.sum().item()
            )

            sample_data = {
                "image": image,
                "mask": mask,
                "probability": probability,
                "prediction": prediction,
                "ground_truth_pixels":
                    ground_truth_pixels,
                "predicted_pixels":
                    predicted_pixels,
                "max_probability": float(
                    probability.max().item()
                ),
                "batch_index":
                    batch_index,
                "sample_index":
                    sample_index,
            }

            if ground_truth_pixels == 0:

                empty_samples.append({
                    **sample_data,
                    "dice": None,
                    "mean_probability_on_tumor":
                        None,
                })

                continue

            intersection = float(
                (
                    prediction
                    * mask
                ).sum().item()
            )

            dice = (
                2.0 * intersection
                + 1e-7
            ) / (
                predicted_pixels
                + ground_truth_pixels
                + 1e-7
            )

            mean_probability_on_tumor = (
                float(
                    probability[
                        mask > 0
                    ]
                    .mean()
                    .item()
                )
            )

            tumor_samples.append({
                **sample_data,
                "dice": float(dice),
                "mean_probability_on_tumor":
                    mean_probability_on_tumor,
            })

    if not tumor_samples:
        raise RuntimeError(
            "No tumor-containing samples found."
        )

    tumor_samples = sorted(
        tumor_samples,
        key=lambda item: item["dice"],
    )

    empty_samples = sorted(
        empty_samples,
        key=lambda item: (
            item["predicted_pixels"],
            item["max_probability"],
        ),
    )

    worst = tumor_samples[0]

    typical = tumor_samples[
        len(tumor_samples) // 2
    ]

    best = tumor_samples[-1]

    selected_samples = [
        {
            **worst,
            "display_name":
                "Worst tumor case",
        },
        {
            **typical,
            "display_name":
                "Typical tumor case",
        },
        {
            **best,
            "display_name":
                "Best tumor case",
        },
    ]

    for index, sample in enumerate(
        empty_samples[
            :number_of_empty_samples
        ]
    ):

        selected_samples.append({
            **sample,
            "display_name":
                f"No-tumor case {index + 1}",
        })

    rows = len(selected_samples)

    figure, axes = plt.subplots(
        rows,
        4,
        figsize=(15, 4 * rows),
        constrained_layout=True,
    )

    if rows == 1:
        axes = np.expand_dims(
            axes,
            axis=0,
        )

    summary = []

    for row, sample in enumerate(
        selected_samples
    ):

        image = (
            sample["image"]
            .permute(1, 2, 0)
            .numpy()
        )

        image = (
            image - image.min()
        ) / (
            image.max()
            - image.min()
            + 1e-8
        )

        mask = sample[
            "mask"
        ].numpy()

        probability = sample[
            "probability"
        ].numpy()

        prediction = sample[
            "prediction"
        ].numpy()

        axes[row, 0].imshow(
            image
        )

        axes[row, 0].set_title(
            sample["display_name"]
        )

        axes[row, 1].imshow(
            mask,
            cmap="gray",
            vmin=0,
            vmax=1,
        )

        axes[row, 1].set_title(
            "Ground truth"
        )

        axes[row, 2].imshow(
            probability,
            cmap="viridis",
            vmin=0,
            vmax=1,
        )

        axes[row, 2].set_title(
            "Probability map"
        )

        axes[row, 3].imshow(
            prediction,
            cmap="gray",
            vmin=0,
            vmax=1,
        )

        if sample["dice"] is None:

            title = (
                "Prediction\n"
                f"False-positive pixels="
                f"{sample['predicted_pixels']}"
            )

        else:

            title = (
                "Prediction\n"
                f"Dice="
                f"{sample['dice']:.4f}"
            )

        axes[row, 3].set_title(
            title
        )

        for axis in axes[row]:
            axis.axis("off")

        summary.append({
            "case":
                sample["display_name"],

            "dice":
                sample["dice"],

            "ground_truth_pixels":
                sample[
                    "ground_truth_pixels"
                ],

            "predicted_pixels":
                sample[
                    "predicted_pixels"
                ],

            "max_probability":
                sample[
                    "max_probability"
                ],

            "mean_probability_on_tumor":
                sample[
                    "mean_probability_on_tumor"
                ],
        })

    figure.suptitle(
        (
            "Representative Test Predictions "
            f"(threshold={threshold:.2f})"
        ),
        fontsize=16,
    )

    plt.show()

    return pd.DataFrame(
        summary
    )


# ---------------------------------------------------------
# Multi-seed evaluation
# ---------------------------------------------------------

FINAL_SEEDS = [
    42,
    123,
    2026,
]


def run_multiple_seeds(
    experiment: ExperimentConfig,
    config: GlobalConfig,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seeds: List[int] = FINAL_SEEDS,
) -> pd.DataFrame:

    results = []

    for seed in seeds:

        seeded_config = replace(
            config,
            seed=seed,
        )

        seeded_experiment = replace(
            experiment,
            name=(
                f"{experiment.name}"
                f"_seed_{seed}"
            ),
        )

        result = train_experiment(
            experiment=seeded_experiment,
            config=seeded_config,
            train_df=train_df,
            validation_df=validation_df,
            test_df=test_df,
        )

        result["seed"] = seed

        results.append(
            result
        )

    results = pd.DataFrame(
        results
    )

    results.to_csv(
        Path(config.results_dir)
        / (
            f"{experiment.name}"
            "_multi_seed.csv"
        ),
        index=False,
    )

    return results


def summarize_multi_seed_results(
    results: pd.DataFrame,
) -> pd.DataFrame:

    metric_columns = [
        "test_dice_tumor",
        "test_iou_tumor",
        "test_recall_tumor",
        "test_specificity_all",
        "test_empty_false_positive_rate",
    ]

    summary = []

    for metric in metric_columns:

        if metric not in results.columns:
            continue

        mean_value = (
            results[metric].mean()
        )

        standard_deviation = (
            results[metric]
            .std(ddof=1)
        )

        summary.append({
            "metric":
                metric,

            "mean":
                mean_value,

            "standard_deviation":
                standard_deviation,

            "formatted":
                (
                    f"{mean_value:.4f} ± "
                    f"{standard_deviation:.4f}"
                ),
        })

    return pd.DataFrame(
        summary
    )


# ---------------------------------------------------------
# Validation threshold optimization
# ---------------------------------------------------------

@torch.no_grad()
def find_recall_oriented_threshold(
    model: nn.Module,
    loader: DataLoader,
    config: GlobalConfig,
    thresholds: Optional[
        np.ndarray
    ] = None,
    beta: float = 2.0,
) -> pd.DataFrame:

    if thresholds is None:

        thresholds = np.arange(
            0.05,
            0.81,
            0.05,
        )

    model.eval()

    all_probabilities = []
    all_masks = []

    for batch in loader:

        images = batch["image"].to(
            config.device
        )

        masks = batch["mask"].to(
            config.device
        )

        outputs = model(
            images
        )

        if isinstance(
            outputs,
            dict,
        ):
            logits = outputs["main"]

        else:
            logits = outputs

        probabilities = torch.sigmoid(
            logits
        )

        all_probabilities.append(
            probabilities
            .detach()
            .cpu()
        )

        all_masks.append(
            masks
            .detach()
            .cpu()
        )

    probabilities = torch.cat(
        all_probabilities,
        dim=0,
    )

    masks = torch.cat(
        all_masks,
        dim=0,
    )

    results = []

    for threshold in thresholds:

        predictions = (
            probabilities
            >= threshold
        ).float()

        true_positive = (
            predictions
            * masks
        ).sum().item()

        false_positive = (
            predictions
            * (1.0 - masks)
        ).sum().item()

        false_negative = (
            (1.0 - predictions)
            * masks
        ).sum().item()

        precision = (
            true_positive
            / (
                true_positive
                + false_positive
                + 1e-8
            )
        )

        recall = (
            true_positive
            / (
                true_positive
                + false_negative
                + 1e-8
            )
        )

        dice = (
            2.0 * true_positive
            / (
                2.0 * true_positive
                + false_positive
                + false_negative
                + 1e-8
            )
        )

        beta_squared = (
            beta ** 2
        )

        f_beta = (
            (1 + beta_squared)
            * precision
            * recall
            / (
                beta_squared
                * precision
                + recall
                + 1e-8
            )
        )

        results.append({
            "threshold":
                float(threshold),

            "precision":
                precision,

            "recall":
                recall,

            "dice":
                dice,

            "f_beta":
                f_beta,

            "false_positive_pixels":
                int(false_positive),

            "false_negative_pixels":
                int(false_negative),
        })

    return pd.DataFrame(
        results
    )


def select_best_threshold(
    threshold_results: pd.DataFrame,
) -> float:
    """
    Select the threshold with the highest F-beta score.
    """

    best_row = (
        threshold_results
        .sort_values(
            "f_beta",
            ascending=False,
        )
        .iloc[0]
    )

    return float(
        best_row["threshold"]
    )


# ---------------------------------------------------------
# Sample-level error analysis
# ---------------------------------------------------------

@torch.no_grad()
def collect_sample_error_analysis(
    model: nn.Module,
    loader: DataLoader,
    config: GlobalConfig,
    threshold: float,
) -> pd.DataFrame:

    model.eval()

    records = []
    dataset_index = 0

    for batch_index, batch in enumerate(
        tqdm(
            loader,
            desc="Collecting errors",
        )
    ):

        images = batch["image"].to(
            config.device
        )

        masks = batch["mask"].to(
            config.device
        )

        outputs = model(
            images
        )

        if isinstance(
            outputs,
            dict,
        ):
            logits = outputs["main"]
        else:
            logits = outputs

        probabilities = torch.sigmoid(
            logits
        )

        predictions = (
            probabilities >= threshold
        ).float()

        for sample_index in range(
            images.size(0)
        ):

            ground_truth = masks[
                sample_index,
                0,
            ]

            prediction = predictions[
                sample_index,
                0,
            ]

            probability = probabilities[
                sample_index,
                0,
            ]

            true_positive = float(
                (
                    prediction
                    * ground_truth
                )
                .sum()
                .item()
            )

            false_positive = float(
                (
                    prediction
                    * (
                        1.0
                        - ground_truth
                    )
                )
                .sum()
                .item()
            )

            false_negative = float(
                (
                    (
                        1.0
                        - prediction
                    )
                    * ground_truth
                )
                .sum()
                .item()
            )

            true_negative = float(
                (
                    (
                        1.0
                        - prediction
                    )
                    * (
                        1.0
                        - ground_truth
                    )
                )
                .sum()
                .item()
            )

            ground_truth_pixels = int(
                ground_truth
                .sum()
                .item()
            )

            predicted_pixels = int(
                prediction
                .sum()
                .item()
            )

            has_tumor = (
                ground_truth_pixels > 0
            )

            dice = (
                2.0 * true_positive
                + 1e-7
            ) / (
                2.0 * true_positive
                + false_positive
                + false_negative
                + 1e-7
            )

            iou = (
                true_positive
                + 1e-7
            ) / (
                true_positive
                + false_positive
                + false_negative
                + 1e-7
            )

            precision = (
                true_positive
                + 1e-7
            ) / (
                true_positive
                + false_positive
                + 1e-7
            )

            recall = (
                true_positive
                + 1e-7
            ) / (
                true_positive
                + false_negative
                + 1e-7
            )

            specificity = (
                true_negative
                + 1e-7
            ) / (
                true_negative
                + false_positive
                + 1e-7
            )

            record = {
                "dataset_index":
                    dataset_index,

                "batch_index":
                    batch_index,

                "sample_index":
                    sample_index,

                "has_tumor":
                    has_tumor,

                "ground_truth_pixels":
                    ground_truth_pixels,

                "predicted_pixels":
                    predicted_pixels,

                "true_positive":
                    int(true_positive),

                "false_positive":
                    int(false_positive),

                "false_negative":
                    int(false_negative),

                "true_negative":
                    int(true_negative),

                "dice":
                    float(dice),

                "iou":
                    float(iou),

                "precision":
                    float(precision),

                "recall":
                    float(recall),

                "specificity":
                    float(specificity),

                "maximum_probability":
                    float(
                        probability
                        .max()
                        .item()
                    ),

                "mean_probability_on_tumor":
                    (
                        float(
                            probability[
                                ground_truth > 0
                            ]
                            .mean()
                            .item()
                        )
                        if has_tumor
                        else np.nan
                    ),

                "completely_missed":
                    (
                        has_tumor
                        and
                        predicted_pixels == 0
                    ),
            }

            if "patient" in batch:

                record["patient"] = (
                    batch["patient"][
                        sample_index
                    ]
                )

            if "image_path" in batch:

                record["image_path"] = (
                    batch["image_path"][
                        sample_index
                    ]
                )

            records.append(
                record
            )

            dataset_index += 1

    return pd.DataFrame(
        records
    )


def summarize_error_analysis(
    error_dataframe: pd.DataFrame,
) -> pd.DataFrame:

    tumor = error_dataframe[
        error_dataframe[
            "has_tumor"
        ]
    ]

    empty = error_dataframe[
        ~error_dataframe[
            "has_tumor"
        ]
    ]

    summary = {
        "metric": [
            "Tumor-containing samples",
            "Mean tumor Dice",
            "Median tumor Dice",
            "Mean tumor IoU",
            "Mean tumor Precision",
            "Mean tumor Recall",
            "Mean tumor Specificity",
            "Completely missed tumors",
            "Completely missed tumor rate",
            "Mean false-negative pixels",
            "Mean false-positive pixels",
            "No-tumor samples",
            "No-tumor cases with false positives",
        ],

        "value": [
            len(tumor),

            tumor[
                "dice"
            ].mean(),

            tumor[
                "dice"
            ].median(),

            tumor[
                "iou"
            ].mean(),

            tumor[
                "precision"
            ].mean(),

            tumor[
                "recall"
            ].mean(),

            tumor[
                "specificity"
            ].mean(),

            int(
                tumor[
                    "completely_missed"
                ].sum()
            ),

            tumor[
                "completely_missed"
            ].mean(),

            tumor[
                "false_negative"
            ].mean(),

            tumor[
                "false_positive"
            ].mean(),

            len(empty),

            int(
                (
                    empty[
                        "false_positive"
                    ] > 0
                ).sum()
            ),
        ],
    }

    return pd.DataFrame(
        summary
    )


# ---------------------------------------------------------
# Segmentation Grad-CAM
# ---------------------------------------------------------

class SegmentationGradCAM:
    """
    Grad-CAM for binary segmentation.

    The target score is the mean tumor logit
    inside a selected spatial region.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
    ):

        self.model = model
        self.target_layer = (
            target_layer
        )

        self.activations = None
        self.gradients = None

        self.forward_handle = (
            target_layer
            .register_forward_hook(
                self._save_activations
            )
        )

        self.backward_handle = (
            target_layer
            .register_full_backward_hook(
                self._save_gradients
            )
        )


    def _save_activations(
        self,
        module,
        inputs,
        output,
    ) -> None:

        if isinstance(
            output,
            (tuple, list),
        ):
            output = output[0]

        self.activations = output


    def _save_gradients(
        self,
        module,
        grad_input,
        grad_output,
    ) -> None:

        if not grad_output:
            self.gradients = None
            return

        self.gradients = (
            grad_output[0]
        )


    @staticmethod
    def _extract_main_logits(
        outputs,
    ) -> torch.Tensor:

        if isinstance(
            outputs,
            dict,
        ):
            return outputs["main"]

        if isinstance(
            outputs,
            (tuple, list),
        ):
            return outputs[0]

        if isinstance(
            outputs,
            torch.Tensor,
        ):
            return outputs

        raise TypeError(
            "Unsupported model output."
        )


    @staticmethod
    def _prepare_region_mask(
        region_mask: torch.Tensor,
        logits: torch.Tensor,
    ) -> torch.Tensor:

        region_mask = torch.as_tensor(
            region_mask,
            device=logits.device,
            dtype=logits.dtype,
        )

        if region_mask.ndim == 2:

            region_mask = (
                region_mask[
                    None,
                    None,
                    :,
                    :,
                ]
            )

        elif region_mask.ndim == 3:

            region_mask = (
                region_mask[:, None]
            )

        if (
            region_mask.shape[-2:]
            != logits.shape[-2:]
        ):

            region_mask = F.interpolate(
                region_mask,
                size=logits.shape[-2:],
                mode="nearest",
            )

        region_mask = (
            region_mask > 0
        ).to(
            logits.dtype
        )

        if (
            region_mask.sum().item()
            <= 0
        ):
            raise ValueError(
                "Grad-CAM target region is empty."
            )

        return region_mask


    def generate(
        self,
        image_tensor: torch.Tensor,
        region_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        self.model.eval()

        device = next(
            self.model.parameters()
        ).device

        image_tensor = (
            image_tensor
            .to(device)
            .contiguous()
        )

        self.model.zero_grad(
            set_to_none=True
        )

        self.activations = None
        self.gradients = None

        with torch.enable_grad():

            outputs = self.model(
                image_tensor
            )

            logits = (
                self._extract_main_logits(
                    outputs
                )
            )

            region_mask = (
                self._prepare_region_mask(
                    region_mask,
                    logits,
                )
            )

            target_score = (
                (
                    logits
                    * region_mask
                ).sum()
                /
                (
                    region_mask.sum()
                    + 1e-8
                )
            )

            target_score.backward()

        if (
            self.activations is None
            or self.gradients is None
        ):
            raise RuntimeError(
                "Grad-CAM hooks failed."
            )

        if (
            self.activations.shape[1]
            <= 1
        ):
            raise ValueError(
                "Grad-CAM requires a "
                "multi-channel feature layer."
            )

        channel_weights = (
            self.gradients.mean(
                dim=(2, 3),
                keepdim=True,
            )
        )

        cam = (
            channel_weights
            * self.activations
        ).sum(
            dim=1,
            keepdim=True,
        )

        cam = F.relu(
            cam
        )

        cam = F.interpolate(
            cam,
            size=image_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        minimum = cam.amin(
            dim=(2, 3),
            keepdim=True,
        )

        maximum = cam.amax(
            dim=(2, 3),
            keepdim=True,
        )

        cam = (
            cam - minimum
        ) / (
            maximum
            - minimum
            + 1e-8
        )

        return {
            "cam":
                cam.detach(),

            "logits":
                logits.detach(),

            "target_score":
                target_score.detach(),
        }


    def remove_hooks(
        self,
    ) -> None:

        if self.forward_handle:
            self.forward_handle.remove()

        if self.backward_handle:
            self.backward_handle.remove()

        self.forward_handle = None
        self.backward_handle = None


# ---------------------------------------------------------
# Select Grad-CAM layer
# ---------------------------------------------------------

def select_gradcam_layer(
    model: nn.Module,
):

    modules = dict(
        model.named_modules()
    )

    preferred_layers = [
        "decoder2.features.main.3",
        "decoder3.features.main.3",
        "decoder1.features.main.3",
    ]

    for name in preferred_layers:

        if name not in modules:
            continue

        layer = modules[name]

        if (
            isinstance(
                layer,
                nn.Conv2d,
            )
            and layer.out_channels > 1
        ):
            return (
                name,
                layer,
            )

    raise RuntimeError(
        "No suitable multi-channel "
        "decoder layer found for Grad-CAM."
    )


# ---------------------------------------------------------
# Grad-CAM visualization helpers
# ---------------------------------------------------------

def normalize_image_for_display(
    image_tensor: torch.Tensor,
) -> np.ndarray:

    image = (
        image_tensor
        .detach()
        .cpu()
        .permute(1, 2, 0)
        .numpy()
    )

    image = (
        image - image.min()
    ) / (
        image.max()
        - image.min()
        + 1e-8
    )

    return image.astype(
        np.float32
    )


def apply_heatmap_overlay(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:

    heatmap_rgb = (
        plt.cm.jet(
            heatmap
        )[..., :3]
    )

    overlay = (
        (1.0 - alpha)
        * image
        +
        alpha
        * heatmap_rgb
    )

    return np.clip(
        overlay,
        0.0,
        1.0,
    )


def generate_gradcam_for_sample(
    model: nn.Module,
    dataset: Dataset,
    gradcam: SegmentationGradCAM,
    dataset_index: int,
    config: GlobalConfig,
    target_mode: str = "ground_truth",
    threshold: float = 0.5,
) -> Dict:

    sample = dataset[
        int(dataset_index)
    ]

    image_tensor = (
        sample["image"]
        .unsqueeze(0)
        .to(config.device)
    )

    ground_truth_tensor = (
        sample["mask"][0]
        .detach()
        .cpu()
    )

    with torch.no_grad():

        outputs = model(
            image_tensor
        )

        if isinstance(
            outputs,
            dict,
        ):
            logits = outputs["main"]
        else:
            logits = outputs

        probability_tensor = (
            torch.sigmoid(
                logits
            )[0, 0]
        )

        prediction_tensor = (
            probability_tensor
            >= threshold
        ).float()

    if target_mode == "ground_truth":

        region_mask = (
            ground_truth_tensor
        )

        if (
            region_mask.sum().item()
            == 0
        ):
            raise ValueError(
                "This sample has no "
                "ground-truth tumor."
            )

    elif target_mode == "prediction":

        region_mask = (
            prediction_tensor
            .detach()
            .cpu()
        )

        if (
            region_mask.sum().item()
            == 0
        ):

            probability_cpu = (
                probability_tensor
                .detach()
                .cpu()
            )

            percentile = (
                torch.quantile(
                    probability_cpu
                    .flatten(),
                    0.95,
                )
            )

            region_mask = (
                probability_cpu
                >= percentile
            ).float()

    else:

        raise ValueError(
            "target_mode must be "
            "'ground_truth' or 'prediction'."
        )

    gradcam_result = (
        gradcam.generate(
            image_tensor=image_tensor,
            region_mask=region_mask,
        )
    )

    cam = (
        gradcam_result["cam"][
            0,
            0,
        ]
        .cpu()
        .numpy()
    )

    image = (
        normalize_image_for_display(
            sample["image"]
        )
    )

    ground_truth = (
        ground_truth_tensor
        .numpy()
    )

    probability = (
        probability_tensor
        .detach()
        .cpu()
        .numpy()
    )

    prediction = (
        prediction_tensor
        .detach()
        .cpu()
        .numpy()
    )

    overlay = (
        apply_heatmap_overlay(
            image,
            cam,
        )
    )

    intersection = float(
        (
            prediction
            * ground_truth
        ).sum()
    )

    dice = (
        2.0 * intersection
        + 1e-7
    ) / (
        prediction.sum()
        + ground_truth.sum()
        + 1e-7
    )

    false_negative = (
        (ground_truth == 1)
        & (prediction == 0)
    ).astype(
        np.float32
    )

    false_positive = (
        (ground_truth == 0)
        & (prediction == 1)
    ).astype(
        np.float32
    )

    return {
        "dataset_index":
            int(dataset_index),

        "patient":
            sample.get(
                "patient",
                "unknown",
            ),

        "image_path":
            sample.get(
                "image_path",
                "",
            ),

        "image":
            image,

        "ground_truth":
            ground_truth,

        "probability":
            probability,

        "prediction":
            prediction,

        "gradcam":
            cam,

        "overlay":
            overlay,

        "false_negative":
            false_negative,

        "false_positive":
            false_positive,

        "dice":
            float(dice),

        "target_mode":
            target_mode,
    }


def show_gradcam_result(
    result: Dict,
) -> None:

    figure, axes = plt.subplots(
        2,
        4,
        figsize=(16, 8),
        constrained_layout=True,
    )

    axes[0, 0].imshow(
        result["image"]
    )
    axes[0, 0].set_title(
        "MRI image"
    )

    axes[0, 1].imshow(
        result["ground_truth"],
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    axes[0, 1].set_title(
        "Ground truth"
    )

    axes[0, 2].imshow(
        result["probability"],
        cmap="viridis",
        vmin=0,
        vmax=1,
    )
    axes[0, 2].set_title(
        "Tumor probability"
    )

    axes[0, 3].imshow(
        result["prediction"],
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    axes[0, 3].set_title(
        (
            "Prediction\n"
            f"Dice={result['dice']:.4f}"
        )
    )

    axes[1, 0].imshow(
        result["gradcam"],
        cmap="jet",
        vmin=0,
        vmax=1,
    )
    axes[1, 0].set_title(
        "Grad-CAM"
    )

    axes[1, 1].imshow(
        result["overlay"]
    )
    axes[1, 1].set_title(
        "Grad-CAM overlay"
    )

    axes[1, 2].imshow(
        result["false_negative"],
        cmap="Reds",
        vmin=0,
        vmax=1,
    )
    axes[1, 2].set_title(
        (
            "False negatives\n"
            f"Pixels="
            f"{int(result['false_negative'].sum())}"
        )
    )

    axes[1, 3].imshow(
        result["false_positive"],
        cmap="Blues",
        vmin=0,
        vmax=1,
    )
    axes[1, 3].set_title(
        (
            "False positives\n"
            f"Pixels="
            f"{int(result['false_positive'].sum())}"
        )
    )

    for axis in axes.ravel():
        axis.axis("off")

    figure.suptitle(
        (
            f"Patient: {result['patient']} | "
            f"Target: {result['target_mode']}"
        )
    )

    plt.show()


# ---------------------------------------------------------
# Representative error cases
# ---------------------------------------------------------

def representative_case_indices(
    error_dataframe: pd.DataFrame,
) -> Dict[str, int]:

    tumor_cases = (
        error_dataframe[
            error_dataframe[
                "has_tumor"
            ]
        ]
        .sort_values(
            "dice",
            ascending=True,
        )
        .reset_index(drop=True)
    )

    if tumor_cases.empty:
        raise ValueError(
            "No tumor cases found."
        )

    return {
        "worst":
            int(
                tumor_cases.iloc[0][
                    "dataset_index"
                ]
            ),

        "typical":
            int(
                tumor_cases.iloc[
                    len(tumor_cases) // 2
                ][
                    "dataset_index"
                ]
            ),

        "best":
            int(
                tumor_cases.iloc[-1][
                    "dataset_index"
                ]
            ),
    }