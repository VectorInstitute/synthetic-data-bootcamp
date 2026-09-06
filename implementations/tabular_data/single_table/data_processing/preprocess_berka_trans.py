#!/usr/bin/env python3
"""Preprocess the Berka (PKDD'99) transaction table.

1. Drop ID columns (``trans_id``, ``account_id``).
2. Convert ``date`` (YYMMDD) to ``trans_date`` = days since the earliest date
   in the loaded data (or a fixed epoch if ``--date-epoch`` is set).
3. Rename ``type`` → ``trans_type``.
4. Fill missing partner ``account`` with 0; fill missing categorical cells with "".
5. Ordinally encode categoricals with ``sklearn.preprocessing.LabelEncoder``
   (sorted unique labels → 0..K-1). Discrete columns:
   ``trans_type``, ``operation``, ``k_symbol``, ``bank``.
6. Optionally subsample, then split into train / holdout (no test set).
7. Write CSV outputs plus ``trans_domain.json``, ``dataset_meta.json``,
   ``meta_info.json``, and encoding artifacts for inverse transforms.


Example 
------------------
from preprocess_berka_trans import preprocess_berka_trans

preprocess_berka_trans(
    input_path="path/to/trans.asc",
    output_dir="./data",
    sample_size=20000,
    holdout_ratio=0.2,
    seed=42,
)
"""

from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Columns in the MIDST / ClavaDDPM single-table Berka export
OUTPUT_COLUMNS = [
    "trans_date",
    "trans_type",
    "operation",
    "amount",
    "balance",
    "k_symbol",
    "bank",
    "account",
]
DISCRETE_COLUMNS = ["trans_type", "operation", "k_symbol", "bank"]
CONTINUOUS_COLUMNS = ["trans_date", "amount", "balance", "account"]
ID_COLUMNS = ["trans_id", "account_id"]

# MIDST evaluation meta: column indices in OUTPUT_COLUMNS order
DEFAULT_META_INFO = {
    "num_col_idx": [0, 3, 4, 7],
    "cat_col_idx": [1, 2, 5, 6],
    "target_col_idx": [1],
    "task_type": "multiclass",
}

DEFAULT_DATASET_META = {
    "relation_order": [[None, "trans"]],
    "tables": {"trans": {"children": [], "parents": []}},
}




def detect_separator(path: Path, explicit: str | None) -> str:
    if explicit is not None:
        return explicit
    if path.suffix.lower() == ".asc":
        return ";"
    # Peek at first line
    with path.open("r", encoding="utf-8", errors="replace") as f:
        header = f.readline()
    if header.count(";") >= header.count(","):
        return ";"
    return ","


def load_raw_transactions(path: Path, sep: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=sep,
        quotechar='"',
        low_memory=False,
    )
    # Numbers / export tools sometimes leave quotes or spaces in headers
    df.columns = [c.strip().strip('"').lower() for c in df.columns]
    required = {
        "date",
        "type",
        "operation",
        "amount",
        "balance",
        "k_symbol",
        "bank",
        "account",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")
    return df


def normalize_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaNs for categoricals with ""; partner account with 0.

    Does **not** strip categorical strings: Berka sometimes uses a single space
    ``" "`` as its own ``k_symbol`` level (9 discrete values including ``""``).
    """
    out = df.copy()
    for col in ["type", "operation", "k_symbol", "bank"]:
        out[col] = out[col].fillna("").astype(str)
        out.loc[out[col].isin(["nan", "None", "<NA>"]), col] = ""

    # Partner account: empty → 0 (matches MIDST preprocessed CSVs)
    out["account"] = pd.to_numeric(out["account"], errors="coerce").fillna(0).astype("int64")
    out["amount"] = pd.to_numeric(out["amount"], errors="coerce")
    out["balance"] = pd.to_numeric(out["balance"], errors="coerce")
    if out["amount"].isna().any() or out["balance"].isna().any():
        raise ValueError("Found non-numeric amount/balance values after parsing.")
    return out


def dates_to_day_offsets(
    date_series: pd.Series,
    epoch_yymmdd: str | None = None,
) -> tuple[pd.Series, str]:
    """Convert YYMMDD integers/strings to days since earliest (or fixed) epoch.

    Mirrors MIDSTModels ``calculate_days_since_earliest_date``.
    """
    as_str = date_series.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    date_objects = [datetime.strptime(d, "%y%m%d") for d in as_str]

    if epoch_yymmdd is None:
        earliest = min(date_objects)
        epoch_str = earliest.strftime("%y%m%d")
    else:
        epoch_str = str(epoch_yymmdd).zfill(6)
        earliest = datetime.strptime(epoch_str, "%y%m%d")

    offsets = pd.Series([(d - earliest).days for d in date_objects], index=date_series.index)
    return offsets, epoch_str


def reconstruct_dates(days_since: pd.Series | list[int], earliest_date_str: str) -> list[str]:
    earliest = datetime.strptime(earliest_date_str, "%y%m%d")
    return [(earliest + timedelta(days=int(d))).strftime("%y%m%d") for d in days_since]


def build_feature_frame(raw: pd.DataFrame, epoch_yymmdd: str | None) -> tuple[pd.DataFrame, str]:
    work = normalize_missing(raw)
    drop_cols = [c for c in ID_COLUMNS if c in work.columns]
    work = work.drop(columns=drop_cols)

    trans_date, epoch = dates_to_day_offsets(work["date"], epoch_yymmdd)
    features = pd.DataFrame(
        {
            "trans_date": trans_date.astype("int64"),
            "trans_type": work["type"].astype(str),
            "operation": work["operation"].astype(str),
            "amount": work["amount"].astype("float64"),
            "balance": work["balance"].astype("float64"),
            "k_symbol": work["k_symbol"].astype(str),
            "bank": work["bank"].astype(str),
            "account": work["account"].astype("int64"),
        }
    )
    return features, epoch


def label_encode(
    df: pd.DataFrame,
    discrete_cols: list[str],
    encoders: dict[str, LabelEncoder] | None = None,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """Fit (optional) and apply LabelEncoders — same as MIDST ``table_label_encode``."""
    out = df.copy()
    fitted: dict[str, LabelEncoder] = {} if encoders is None else dict(encoders)

    for col in discrete_cols:
        if encoders is None:
            le = LabelEncoder()
            out[col] = le.fit_transform(out[col].astype(str))
            fitted[col] = le
        else:
            le = encoders[col]
            # Map unseen labels to -1 then shift unknowns to max+1 if needed
            known = set(le.classes_)
            values = out[col].astype(str)
            unseen = ~values.isin(known)
            if unseen.any():
                raise ValueError(
                    f"Column {col!r} has labels not seen during fit: "
                    f"{sorted(values[unseen].unique())[:10]}"
                )
            out[col] = le.transform(values)
        out[col] = out[col].astype("int64")
    return out, fitted


def get_domain(df: pd.DataFrame, discrete_cols: list[str]) -> dict[str, dict[str, str]]:
    """Build a type-only ``trans_domain.json`` (IDs already dropped)."""
    domain: dict[str, dict[str, str]] = {}
    for col in df.columns:
        domain[col] = {"type": "discrete" if col in discrete_cols else "continuous"}
    return domain


def sample_dataframe(df: pd.DataFrame, sample_size: int | None, seed: int) -> pd.DataFrame:
    if sample_size is None:
        return df
    if sample_size <= 0:
        raise ValueError("--sample-size must be positive")
    if sample_size >= len(df):
        print(f"sample-size {sample_size} >= nrows {len(df)}; using full table.")
        return df
    return df.sample(n=sample_size, random_state=seed).reset_index(drop=True)


def split_train_holdout(
    df: pd.DataFrame, holdout_ratio: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < holdout_ratio < 1.0:
        raise ValueError("--holdout-ratio must be in (0, 1)")
    train_df, holdout_df = train_test_split(
        df,
        test_size=holdout_ratio,
        random_state=seed,
        shuffle=True,
    )
    return train_df.reset_index(drop=True), holdout_df.reset_index(drop=True)


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=4) + "\n", encoding="utf-8")


def preprocess_berka_trans(
    input_path: Path | str,
    output_dir: Path | str = Path("data"),
    sep: str | None = None,
    sample_size: int | None = None,
    sample_before_encode: bool = True,
    holdout_ratio: float = 0.2,
    seed: int = 42,
    date_epoch: str | None = None,
    save_artifacts: bool = True,
) -> dict[str, Any]:
    """Run the full Berka trans preprocessing pipeline and write outputs.

    Parameters:
    - input_path: Path to the raw data file.
    - output_dir: Path to the output directory.
    - sep: Field separator. Default: auto-detect ('; for .asc, comma otherwise).
    - sample_size: If set, randomly sample this many rows.
    - sample_before_encode: Sample before fitting LabelEncoders. Faster on 1M rows, but category codes may differ from a full-data export.
    - holdout_ratio: Fraction of rows for the holdout set (default: 0.2). Train gets the rest.
    - seed: Random seed for sampling and the train/holdout split.
    - date_epoch: Optional fixed YYMMDD epoch for day offsets (e.g. 930101). Default: earliest date present in the (possibly sampled) data — same as MIDSTModels calculate_days_since_earliest_date.
    - save_artifacts: Whether to save the artifacts (preprocess_meta.json, trans_label_encoders.pkl, trans_domain.json, dataset_meta.json, meta_info.json).
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detected_sep = detect_separator(input_path, sep)
    print(f"Loading {input_path} (sep={detected_sep!r}) ...")
    raw = load_raw_transactions(input_path, detected_sep)
    print(f"Loaded {len(raw):,} rows, {list(raw.columns)}")

    if sample_before_encode and sample_size is not None:
        raw = sample_dataframe(raw, sample_size, seed)
        print(f"Sampled to {len(raw):,} rows before encoding.")

    features, epoch = build_feature_frame(raw, date_epoch)
    print(f"Date epoch (YYMMDD): {epoch}  →  trans_date = days since that date")

    encoded, encoders = label_encode(features, DISCRETE_COLUMNS)
    print("LabelEncoder classes:")
    for col, le in encoders.items():
        print(f"  {col}: {list(le.classes_)}")

    domain = get_domain(encoded, DISCRETE_COLUMNS)

    if not sample_before_encode:
        encoded = sample_dataframe(encoded, sample_size, seed)
        if sample_size is not None:
            print(f"Sampled to {len(encoded):,} rows after encoding.")

    encoded = encoded[OUTPUT_COLUMNS]
    train_df, holdout_df = split_train_holdout(encoded, holdout_ratio, seed)
    print(
        f"Split: train={len(train_df):,}  holdout={len(holdout_df):,}  "
        f"(holdout_ratio={holdout_ratio})"
    )

    # --- writes ---
    train_path = output_dir / "trans.csv"
    holdout_path = output_dir / "trans_holdout.csv"
    train_df.to_csv(train_path, index=False)
    holdout_df.to_csv(holdout_path, index=False)
    print(f"Wrote train data to {train_path}")
    print(f"Wrote holdout data to {holdout_path}")
    save_json(output_dir / "trans_domain.json", domain)
    save_json(output_dir / "dataset_meta.json", DEFAULT_DATASET_META)
    save_json(output_dir / "meta_info.json", DEFAULT_META_INFO)
    print(f"Wrote {output_dir / 'trans_domain.json'}")
    print(f"Wrote {output_dir / 'dataset_meta.json'}")
    print(f"Wrote {output_dir / 'meta_info.json'}")


    if save_artifacts:
        preprocess_meta = {
            "date_epoch_yymmdd": epoch,
            "discrete_columns": DISCRETE_COLUMNS,
            "continuous_columns": CONTINUOUS_COLUMNS,
            "output_columns": OUTPUT_COLUMNS,
            "n_rows_processed": len(encoded),
            "n_train": len(train_df),
            "n_holdout": len(holdout_df),
            "sample_size": sample_size,
            "sample_before_encode": sample_before_encode,
            "holdout_ratio": holdout_ratio,
            "seed": seed,
            "label_classes": {col: list(le.classes_) for col, le in encoders.items()},
        }
        save_json(output_dir / "preprocess_meta.json", preprocess_meta)
        encoders_path = output_dir / "trans_label_encoders.pkl"
        with encoders_path.open("wb") as f:
            pickle.dump({"encoders": encoders, "date_epoch_yymmdd": epoch}, f)
        print(f"Wrote {output_dir / 'preprocess_meta.json'}")
        print(f"Wrote {encoders_path}")

    print("Done.")

    return {
        "output_dir": output_dir,
        "domain": domain,
        "n_rows": len(encoded),
        "n_train": len(train_df),
        "n_holdout": len(holdout_df),
    }
