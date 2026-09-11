"""Preprocessing helpers for tabular quality, utility, and privacy metrics."""

from typing import Any, overload

import pandas as pd
from midst_toolkit.data_processing.utils import SynthEvalDataframeEncoding
from midst_toolkit.evaluation.utils import (
    extract_columns_based_on_meta_info,
    one_hot_encode_categoricals_and_merge_with_numerical,
)


def preprocess_data_for_alpha_precision_eval(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    meta_info: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare real and synthetic data for Alpha-precision evaluation.

    Categorical columns are one-hot encoded and numerical columns remain
    unchanged when comparing ``real_data`` to ``synthetic_data``.

    This follows the convention of:

    https://github.com/VectorInstitute/MIDSTModels/blob/main/midst_models/single_table_TabDDPM/eval/eval_quality.py

    Args:
        real_data: Real data to which the synthetic data will be compared.
        synthetic_data: Synthetic data whose quality is to be measured.
        meta_info: Metadata used to find numerical and categorical columns.

    Returns
    -------
        Preprocessed real and synthetic dataframes, in that order.
    """
    numerical_real_data, categorical_real_data = extract_columns_based_on_meta_info(real_data, meta_info)
    numerical_synthetic_data, categorical_synthetic_data = extract_columns_based_on_meta_info(
        synthetic_data,
        meta_info,
    )

    numerical_real_numpy, categorical_real_numpy, numerical_synthetic_numpy, categorical_synthetic_numpy = (
        numerical_real_data.to_numpy(),
        categorical_real_data.to_numpy().astype("str"),
        numerical_synthetic_data.to_numpy(),
        categorical_synthetic_data.to_numpy().astype("str"),
    )

    return one_hot_encode_categoricals_and_merge_with_numerical(
        categorical_real_numpy,
        categorical_synthetic_numpy,
        numerical_real_numpy,
        numerical_synthetic_numpy,
    )


def get_numerical_and_categorical_column_names(
    data: pd.DataFrame,
    meta_info: dict[str, Any],
) -> tuple[list[str], list[str]]:
    """Extract numerical and categorical column names from metadata.

    Args:
        data: Collection of data with a set of column names that will be extracted.
        meta_info: Metadata whose column indices mark numerical and categorical
            columns of the provided dataset.

    Returns
    -------
        Names of numerical and categorical columns, respectively.
    """
    # Enumerate columns and replace column name with index
    column_names = list(data.columns)

    # Get numerical and categorical column indices from meta info.
    # These are the only admissible/generate-able column types.
    numerical_column_idx = meta_info["num_col_idx"]
    categorical_column_idx = meta_info["cat_col_idx"]

    return [column_names[i] for i in numerical_column_idx], [column_names[i] for i in categorical_column_idx]


@overload
def syntheval_preprocess(
    numerical_columns: list[str],
    categorical_columns: list[str],
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]: ...


@overload
def syntheval_preprocess(
    numerical_columns: list[str],
    categorical_columns: list[str],
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    holdout_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: ...


def syntheval_preprocess(
    numerical_columns: list[str],
    categorical_columns: list[str],
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    holdout_data: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame]:
    """Encode tables with the SynthEval preprocessing pipeline.

    Numerical columns are min-max encoded and categorical columns are
    ordinally encoded (not one-hot encoded). If holdout data is provided,
    fitting is performed on all three tables.

    Args:
        numerical_columns: Numerical column names in the respective dataframes.
        categorical_columns: Categorical column names in the dataframes.
        real_data: Real data often used to train the generative model.
        synthetic_data: Dataframe containing synthetic data.
        holdout_data: Real data often withheld from training. If None, fitting
            uses only ``real_data`` and ``synthetic_data``. Defaults to None.

    Returns
    -------
        Preprocessed real, synthetic, and possibly holdout dataframes.
    """
    encoder = SynthEvalDataframeEncoding(
        real_data,
        synthetic_data,
        categorical_columns,
        numerical_columns,
        holdout_data=holdout_data,
    )
    real_data = encoder.encode(real_data)
    synthetic_data = encoder.encode(synthetic_data)

    if holdout_data is not None:
        return real_data, synthetic_data, encoder.encode(holdout_data)
    return real_data, synthetic_data


def remove_label_column_from_other_columns(
    label_column: str,
    numerical_columns: list[str],
    categorical_columns: list[str],
) -> tuple[list[str], list[str]]:
    """Remove a task label from numerical and categorical column lists.

    During preprocessing it is useful to encode the label column, for example
    as an ordinal value. For F1 measurements the label should no longer be
    part of the feature dataframe during training.

    Args:
        label_column: Column name associated with task labels of interest.
        numerical_columns: Column names associated with numerical values.
        categorical_columns: Column names associated with categorical values.

    Raises
    ------
        ValueError: If the label column appears in both column-name lists.

    Returns
    -------
        Numerical and categorical names without the specified label column.
    """
    if label_column in numerical_columns and label_column in categorical_columns:
        raise ValueError("Label column appears in both types of columns...")
    return [item for item in numerical_columns if label_column != item], [
        item for item in categorical_columns if label_column != item
    ]
