
from pathlib import Path
import os

# Make sure we are in the root directory
def set_project_root(marker="pyproject.toml"):
    path = Path.cwd()
    for parent in [path, *path.parents]:
        if (parent / marker).exists():
            os.chdir(parent)
            return parent
    raise FileNotFoundError(f"Could not find {marker} in any parent directory")
set_project_root()

from logging import INFO
from typing import Any

from hydra import initialize, compose
import pandas as pd
from omegaconf import DictConfig, OmegaConf
import json

from midst_toolkit.common.logger import log
from midst_toolkit.data_processing.midst_data_processing import load_midst_data_with_test
from midst_toolkit.evaluation.metrics_base import MetricBase

from midst_toolkit.evaluation.privacy.distance_preprocess import preprocess_for_distance_computation
from midst_toolkit.evaluation.privacy.distance_utils import NormType
from midst_toolkit.evaluation.privacy.epsilon_identifiability_risk import EpsilonIdentifiabilityNorm
from midst_toolkit.evaluation.quality import (
    MeanRegressionDifference,
)
from midst_toolkit.evaluation.quality.confidence_interval_overlap import ConfidenceLevel

# Local imports
from implementations.tabular_data.evaluation.preprocessing import (   
    get_numerical_and_categorical_column_names,
    preprocess_data_for_alpha_precision_eval,
    syntheval_preprocess,
)
from implementations.tabular_data.evaluation.display_utils import log_metrics


from midst_toolkit.evaluation.quality import MeanRegressionDifference
from pathlib import Path
from implementations.tabular_data.evaluation.preprocessing import remove_label_column_from_other_columns

def run_regression_score_difference(numerical_columns, categorical_columns, syntheval_real_data_train, syntheval_synthetic_data, syntheval_real_data_holdout):
    TASK_TYPE = "regression"
    # Make sure that the label column is a numerical column
    LABEL_COLUMN = "balance"
    # Specify the regression models' parameters and structure in a config file. Make sure it's a PATH object.
    REGRESSOIN_CONFIG_PATH = Path("implementations/tabular_data/evaluation/regression_config.yaml")
    # Explicitly removing the target/label column from other column names
    filtered_numerical_columns, filtered_categorical_columns = remove_label_column_from_other_columns(
        LABEL_COLUMN, numerical_columns, categorical_columns
    )
    metric = MeanRegressionDifference(
        categorical_columns=filtered_categorical_columns,
        numerical_columns=filtered_numerical_columns,
        label_column=LABEL_COLUMN,
        preprocess_labels=True,
        include_additional_metrics=True,
        # Regression has it's own preprocessing pipeline
        do_preprocess=True,
        regressors_config = REGRESSOIN_CONFIG_PATH,
        measure_metrics_in_original_label_space=False,
    )
    results = metric.compute(syntheval_real_data_train, syntheval_synthetic_data, syntheval_real_data_holdout)
    return results

if __name__ == "__main__":
    # Make sure we are in the root directory
    def set_project_root(marker="pyproject.toml"):
        path = Path.cwd()
        for parent in [path, *path.parents]:
            if (parent / marker).exists():
                os.chdir(parent)
                return parent
        raise FileNotFoundError(f"Could not find {marker} in any parent directory")
    set_project_root()
    ROOT = Path.cwd()
    IMPLEMENTATION_ROOT = ROOT / "implementations" / "tabular_data" / "single_table" 
    # Set data and output directories
    base_data_dir = IMPLEMENTATION_ROOT / "data"
    base_output_dir = IMPLEMENTATION_ROOT / "results"

    TABLE_NAME = "trans"

    real_train_data = pd.read_csv(base_data_dir / f"{TABLE_NAME}.csv")
    real_holdout_data = pd.read_csv(base_data_dir / f"{TABLE_NAME}_holdout.csv")
    synthetic_data = pd.read_csv(base_output_dir / f"{TABLE_NAME}_synthetic.csv")

    with open(base_data_dir / "meta_info.json", "r") as f:
        meta_info = json.load(f)

    log(INFO, f"Loaded {TABLE_NAME} data for evaluations")
    log(INFO, f"Loaded meta_info for {TABLE_NAME} data")
    log(INFO, f"Loaded {len(real_train_data)} rows of real training data")
    log(INFO, f"Loaded {len(real_holdout_data)} rows of real holdout data")
    log(INFO, f"Loaded {len(synthetic_data)} rows of synthetic data")

    # Shared preprocessing for syntheval based metrics if they are to be run
    log(INFO, "Preprocessing Data with SynthEval pipeline")
    numerical_columns, categorical_columns = get_numerical_and_categorical_column_names(real_train_data, meta_info)
    # Categorical values are ordinal encoded, numerical values are min-max scaled
    syntheval_real_data_train, syntheval_synthetic_data, syntheval_real_data_holdout = syntheval_preprocess(
        numerical_columns, categorical_columns, real_train_data, synthetic_data, real_holdout_data
    )
    results = run_regression_score_difference(numerical_columns, categorical_columns, syntheval_real_data_train, syntheval_synthetic_data, syntheval_real_data_holdout)
    print(results)