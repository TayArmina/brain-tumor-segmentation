from dataclasses import replace
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .config import GlobalConfig, ExperimentConfig
from .training import train_experiment


# ---------------------------------------------------------
# Main architecture ablation
# ---------------------------------------------------------

MAIN_EXPERIMENTS = [

    ExperimentConfig(
        name="unet",
        use_residual=False,
        use_cbam=False,
        use_augmentation=False,
        use_aspp=True,
        use_deep_supervision=True,
        loss_name="dice_focal",
    ),

    ExperimentConfig(
        name="unet_aug",
        use_residual=False,
        use_cbam=False,
        use_augmentation=True,
        use_aspp=True,
        use_deep_supervision=True,
        loss_name="dice_focal",
    ),

    ExperimentConfig(
        name="residual_unet",
        use_residual=True,
        use_cbam=False,
        use_augmentation=False,
        use_aspp=True,
        use_deep_supervision=True,
        loss_name="dice_focal",
    ),

    ExperimentConfig(
        name="residual_unet_aug",
        use_residual=True,
        use_cbam=False,
        use_augmentation=True,
        use_aspp=True,
        use_deep_supervision=True,
        loss_name="dice_focal",
    ),

    ExperimentConfig(
        name="cbam_unet",
        use_residual=False,
        use_cbam=True,
        use_augmentation=False,
        use_aspp=True,
        use_deep_supervision=True,
        loss_name="dice_focal",
    ),

    ExperimentConfig(
        name="cbam_unet_aug",
        use_residual=False,
        use_cbam=True,
        use_augmentation=True,
        use_aspp=True,
        use_deep_supervision=True,
        loss_name="dice_focal",
    ),

    ExperimentConfig(
        name="residual_cbam_unet",
        use_residual=True,
        use_cbam=True,
        use_augmentation=False,
        use_aspp=True,
        use_deep_supervision=True,
        loss_name="dice_focal",
    ),

    ExperimentConfig(
        name="residual_cbam_unet_aug",
        use_residual=True,
        use_cbam=True,
        use_augmentation=True,
        use_aspp=True,
        use_deep_supervision=True,
        loss_name="dice_focal",
    ),
]


# ---------------------------------------------------------
# Loss comparison
# ---------------------------------------------------------

TVERSKY_LOSS_EXPERIMENT = ExperimentConfig(
    name="residual_cbam_unet_aug_dice_focal_tversky",
    use_residual=True,
    use_cbam=True,
    use_augmentation=True,
    use_aspp=True,
    use_deep_supervision=True,
    loss_name="dice_focal_tversky",
)


LOSS_COMPARISON_EXPERIMENTS = [
    MAIN_EXPERIMENTS[-1],
    TVERSKY_LOSS_EXPERIMENT,
]


# ---------------------------------------------------------
# Run a group of experiments
# ---------------------------------------------------------

def run_experiment_set(
    experiments: List[ExperimentConfig],
    config: GlobalConfig,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    result_filename: str,
) -> pd.DataFrame:
    """
    Run experiments while skipping experiments
    that have already been completed.
    """

    result_path = (
        Path(config.results_dir)
        / result_filename
    )

    if result_path.exists():

        results = (
            pd.read_csv(result_path)
            .to_dict("records")
        )

    else:
        results = []

    completed_experiments = {
        result["experiment"]
        for result in results
    }

    for experiment in experiments:

        if experiment.name in completed_experiments:

            print(
                "Skipping completed experiment:",
                experiment.name,
            )

            continue

        result = train_experiment(
            experiment=experiment,
            config=config,
            train_df=train_df,
            validation_df=validation_df,
            test_df=test_df,
        )

        results.append(result)

        completed_experiments.add(
            experiment.name
        )

        pd.DataFrame(
            results
        ).to_csv(
            result_path,
            index=False,
        )

    return pd.DataFrame(results)


# ---------------------------------------------------------
# Select best main architecture
# ---------------------------------------------------------

def select_best_main_experiment(
    results: pd.DataFrame,
) -> ExperimentConfig:
    """
    Select the baseline Dice-Focal architecture
    with the highest tumor Dice.
    """

    if "loss_name" not in results.columns:
        results = results.copy()
        results["loss_name"] = "dice_focal"

    baseline_results = results[
        results["loss_name"]
        == "dice_focal"
    ].copy()

    if baseline_results.empty:
        raise ValueError(
            "No Dice-Focal baseline results were found."
        )

    best_row = (
        baseline_results
        .sort_values(
            "test_dice_tumor",
            ascending=False,
        )
        .iloc[0]
    )

    for experiment in MAIN_EXPERIMENTS:

        if (
            experiment.name
            == best_row["experiment"]
        ):
            return experiment

    raise ValueError(
        "Best experiment was not found "
        "in MAIN_EXPERIMENTS."
    )


# ---------------------------------------------------------
# ASPP + Deep Supervision ablation
# ---------------------------------------------------------

def build_secondary_experiments(
    best_experiment: ExperimentConfig,
) -> List[ExperimentConfig]:

    return [

        replace(
            best_experiment,
            name=(
                f"{best_experiment.name}"
                "_no_aspp_no_ds"
            ),
            use_aspp=False,
            use_deep_supervision=False,
        ),

        replace(
            best_experiment,
            name=(
                f"{best_experiment.name}"
                "_aspp_only"
            ),
            use_aspp=True,
            use_deep_supervision=False,
        ),

        replace(
            best_experiment,
            name=(
                f"{best_experiment.name}"
                "_ds_only"
            ),
            use_aspp=False,
            use_deep_supervision=True,
        ),

        replace(
            best_experiment,
            name=(
                f"{best_experiment.name}"
                "_aspp_ds"
            ),
            use_aspp=True,
            use_deep_supervision=True,
        ),
    ]


# ---------------------------------------------------------
# Final comparison table
# ---------------------------------------------------------

def build_final_comparison(
    config: GlobalConfig,
) -> pd.DataFrame:

    result_files = [

        Path(config.results_dir)
        / "main_ablation_results.csv",

        Path(config.results_dir)
        / "aspp_deep_supervision_ablation.csv",
    ]

    tables = []

    for path in result_files:

        if not path.exists():
            continue

        table = pd.read_csv(path)

        if "loss_name" not in table.columns:
            table["loss_name"] = "dice_focal"

        table["loss_name"] = (
            table["loss_name"]
            .fillna("dice_focal")
        )

        tables.append(table)

    if not tables:
        raise FileNotFoundError(
            "No experiment result files were found."
        )

    results = (
        pd.concat(
            tables,
            ignore_index=True,
        )
        .drop_duplicates(
            subset=[
                "experiment",
                "loss_name",
            ],
            keep="last",
        )
    )

    comparison_columns = [
        "experiment",
        "loss_name",
        "residual",
        "cbam",
        "augmentation",
        "aspp",
        "deep_supervision",
        "parameters",
        "best_epoch",
        "training_minutes",
        "test_dice_all",
        "test_dice_tumor",
        "test_dice_empty",
        "test_iou_tumor",
        "test_precision_all",
        "test_recall_tumor",
        "test_specificity_all",
        "test_empty_false_positive_rate",
    ]

    comparison_columns = [
        column
        for column in comparison_columns
        if column in results.columns
    ]

    comparison = (
        results[
            comparison_columns
        ]
        .sort_values(
            [
                "test_dice_tumor",
                "test_recall_tumor",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    comparison.to_csv(
        Path(config.results_dir)
        / "final_comparison.csv",
        index=False,
    )

    return comparison