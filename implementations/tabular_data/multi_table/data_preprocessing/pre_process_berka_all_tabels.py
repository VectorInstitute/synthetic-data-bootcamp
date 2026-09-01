#!/usr/bin/env python3
"""Preprocess all Berka (PKDD'99) tables for ClavaDDPM multi-table training.

Keeps primary/foreign key columns in the CSVs so tables can be joined. Those
columns are omitted from ``{table}_domain.json`` because
``midst_toolkit.models.clavaddpm.data_loaders.load_tables`` drops any column
whose name contains ``_id`` before reading domain types.

Transforms (per table):
1. Convert date columns (YYMMDD) to days since the earliest date in that table
   (or a fixed epoch if ``date_epoch`` is set).
2. Split client ``birth_number`` into year / month / day / gender.
3. Fill missing categoricals with ``""`` (a lone space is kept as its own level).
4. Ordinally encode discrete columns with ``sklearn.preprocessing.LabelEncoder``.

Call ``preprocess_berka_all_tables(input_dir, output_dir)`` from the repo root.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from implementations.tabular_data.single_table.data_processing.preprocess_berka_trans import (
    dates_to_day_offsets,
    detect_separator,
    label_encode,
    sample_dataframe,
)


TABLE_NAMES = [
    "account",
    "card",
    "client",
    "disp",
    "district",
    "loan",
    "order",
    "trans",
]

TABLE_SPECS: dict[str, dict[str, Any]] = {
    "account": {
        "id_columns": ["account_id", "district_id"],
        "discrete_columns": ["frequency"],
        "continuous_columns": ["account_date"],
        "required": ["account_id", "district_id", "frequency", "date"],
    },
    "card": {
        "id_columns": ["card_id", "disp_id"],
        "discrete_columns": ["card_type"],
        "continuous_columns": ["issued"],
        "required": ["card_id", "disp_id", "type", "issued"],
    },
    "client": {
        "id_columns": ["client_id", "district_id"],
        "discrete_columns": ["year", "month", "client_date", "gender"],
        "continuous_columns": [],
        "required": ["client_id", "birth_number", "district_id"],
    },
    "disp": {
        "id_columns": ["disp_id", "client_id", "account_id"],
        "discrete_columns": ["disp_type"],
        "continuous_columns": [],
        "required": ["disp_id", "client_id", "account_id", "type"],
    },
    "district": {
        "id_columns": ["district_id"],
        "discrete_columns": ["A2", "A3"],
        "continuous_columns": [
            "A4",
            "A5",
            "A6",
            "A7",
            "A8",
            "A9",
            "A10",
            "A11",
            "A12",
            "A13",
            "A14",
            "A15",
            "A16",
        ],
        "required": ["A1"] + [f"A{i}" for i in range(2, 17)],
    },
    "loan": {
        "id_columns": ["loan_id", "account_id"],
        "discrete_columns": ["status"],
        "continuous_columns": ["loan_date", "amount", "duration", "payments"],
        "required": ["loan_id", "account_id", "date", "amount", "duration", "payments", "status"],
    },
    "order": {
        "id_columns": ["order_id", "account_id"],
        "discrete_columns": ["bank_to", "k_symbol"],
        "continuous_columns": ["account_to", "amount"],
        "required": ["order_id", "account_id", "bank_to", "account_to", "amount", "k_symbol"],
    },
    "trans": {
        "id_columns": ["trans_id", "account_id"],
        "discrete_columns": ["trans_type", "operation", "k_symbol", "bank"],
        "continuous_columns": ["trans_date", "amount", "balance", "account"],
        "required": [
            "trans_id",
            "account_id",
            "date",
            "type",
            "operation",
            "amount",
            "balance",
            "k_symbol",
            "bank",
            "account",
        ],
    },
}

DEFAULT_DATASET_META = {
    "relation_order": [
        [None, "district"],
        ["district", "client"],
        ["district", "account"],
        ["client", "disp"],
        ["account", "disp"],
        ["disp", "card"],
        ["account", "loan"],
        ["account", "order"],
        ["account", "trans"],
    ],
    "tables": {
        "district": {"children": ["client", "account"], "parents": []},
        "client": {"children": ["disp"], "parents": ["district"]},
        "account": {"children": ["disp", "loan", "order", "trans"], "parents": ["district"]},
        "disp": {"children": ["card"], "parents": ["client", "account"]},
        "card": {"children": [], "parents": ["disp"]},
        "loan": {"children": [], "parents": ["account"]},
        "order": {"children": [], "parents": ["account"]},
        "trans": {"children": [], "parents": ["account"]},
    },
}

_THIS_DIR = Path(__file__).resolve().parent
_DEFAULT_OUTPUT_DIR = _THIS_DIR.parent / "data" / "berka"
_DEFAULT_INPUT_DIR = _DEFAULT_OUTPUT_DIR / "raw_data"


def save_json(path: Path, payload: Any) -> None:
    """Write ``payload`` as indented JSON plus a trailing newline."""
    path.write_text(json.dumps(payload, indent=4) + "\n", encoding="utf-8")


def find_table_path(input_dir: Path, table: str) -> Path:
    """Return the first matching ``{table}.csv`` or ``{table}.asc`` path."""
    for name in (f"{table}.csv", f"{table}.asc", f"{table.upper()}.CSV", f"{table.upper()}.ASC"):
        candidate = input_dir / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No raw file for table {table!r} in {input_dir} "
        f"(expected {table}.csv or {table}.asc).",
    )


def _normalize_columns(columns: list[str], *, district: bool) -> list[str]:
    cleaned = [c.strip().strip('"') for c in columns]
    if district:
        return cleaned
    return [c.lower() for c in cleaned]


def load_raw_table(path: Path, table: str, sep: str | None = None) -> pd.DataFrame:
    """Load one Berka table and normalize column names."""
    detected = detect_separator(path, sep)
    df = pd.read_csv(path, sep=detected, quotechar='"', low_memory=False)
    df.columns = _normalize_columns(list(df.columns), district=(table == "district"))
    if table == "district" and "district_id" in df.columns and "A1" not in df.columns:
        df = df.rename(columns={"district_id": "A1"})
    required = set(TABLE_SPECS[table]["required"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{table} is missing required columns: {sorted(missing)}")
    return df


def fill_categorical(series: pd.Series) -> pd.Series:
    """Fill NaNs with empty string; keep a lone space as its own k_symbol level."""
    out = series.fillna("").astype(str)
    out.loc[out.isin(["nan", "None", "<NA>"])] = ""
    return out


def yymmdd_strings(date_series: pd.Series) -> pd.Series:
    """Normalize YYMMDD values; strip trailing time from card.issued if present."""
    as_str = date_series.astype(str).str.replace(r"\.0$", "", regex=True)
    as_str = as_str.str.replace(r"\s+.*$", "", regex=True)
    return as_str.str.slice(0, 6).str.zfill(6)


def to_day_offsets(date_series: pd.Series, epoch_yymmdd: str | None) -> tuple[pd.Series, str]:
    """Convert YYMMDD values to integer day offsets."""
    return dates_to_day_offsets(yymmdd_strings(date_series), epoch_yymmdd)


def birth_number_split(birth_numbers: pd.Series) -> pd.DataFrame:
    """Split YYMMDD birth numbers; month >= 50 encodes female (ClavaDDPM)."""
    as_str = birth_numbers.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    years = as_str.str.slice(0, 2).astype(int)
    months = as_str.str.slice(2, 4).astype(int)
    days = as_str.str.slice(4, 6).astype(int)
    female = months >= 50
    months = months.where(~female, months - 50)
    gender = female.astype(int)
    return pd.DataFrame(
        {
            "year": years.to_numpy(),
            "month": months.to_numpy(),
            "client_date": days.to_numpy(),
            "gender": gender.to_numpy(),
        },
        index=birth_numbers.index,
    )


def _id_frame(raw: pd.DataFrame, id_columns: list[str]) -> pd.DataFrame:
    missing = [c for c in id_columns if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing ID columns: {missing}")
    return raw[id_columns].copy()


def transform_account(raw: pd.DataFrame, date_epoch: str | None) -> tuple[pd.DataFrame, str]:
    """Encode account frequency and convert date to day offsets."""
    work = raw.copy()
    work["frequency"] = fill_categorical(work["frequency"])
    account_date, epoch = to_day_offsets(work["date"], date_epoch)
    features = pd.DataFrame(
        {
            "frequency": work["frequency"],
            "account_date": account_date.astype("int64"),
        },
        index=work.index,
    )
    out = pd.concat([_id_frame(work, TABLE_SPECS["account"]["id_columns"]), features], axis=1)
    return out, epoch


def transform_card(raw: pd.DataFrame, date_epoch: str | None) -> tuple[pd.DataFrame, str]:
    """Rename card type and convert issued date to day offsets."""
    work = raw.copy()
    issued, epoch = to_day_offsets(work["issued"], date_epoch)
    features = pd.DataFrame(
        {
            "card_type": fill_categorical(work["type"]),
            "issued": issued.astype("int64"),
        },
        index=work.index,
    )
    out = pd.concat([_id_frame(work, TABLE_SPECS["card"]["id_columns"]), features], axis=1)
    return out, epoch


def transform_client(raw: pd.DataFrame, date_epoch: str | None) -> tuple[pd.DataFrame, None]:
    """Split birth_number into year, month, day, and gender."""
    _ = date_epoch
    work = raw.copy()
    split = birth_number_split(work["birth_number"])
    out = pd.concat([_id_frame(work, TABLE_SPECS["client"]["id_columns"]), split], axis=1)
    return out, None


def transform_disp(raw: pd.DataFrame, date_epoch: str | None) -> tuple[pd.DataFrame, None]:
    """Rename disposition type."""
    _ = date_epoch
    work = raw.copy()
    features = pd.DataFrame({"disp_type": fill_categorical(work["type"])}, index=work.index)
    out = pd.concat([_id_frame(work, TABLE_SPECS["disp"]["id_columns"]), features], axis=1)
    return out, None


def transform_district(raw: pd.DataFrame, date_epoch: str | None) -> tuple[pd.DataFrame, None]:
    """Rename A1 to district_id and coerce demographic numerics."""
    _ = date_epoch
    work = raw.copy()
    ids = pd.DataFrame({"district_id": work["A1"]}, index=work.index)
    features = pd.DataFrame(index=work.index)
    features["A2"] = fill_categorical(work["A2"])
    features["A3"] = fill_categorical(work["A3"])
    for col in TABLE_SPECS["district"]["continuous_columns"]:
        numeric = pd.to_numeric(work[col], errors="coerce").fillna(0)
        features[col] = numeric
    out = pd.concat([ids, features], axis=1)
    return out, None


def transform_loan(raw: pd.DataFrame, date_epoch: str | None) -> tuple[pd.DataFrame, str]:
    """Convert loan date to day offsets; keep duration as continuous."""
    work = raw.copy()
    loan_date, epoch = to_day_offsets(work["date"], date_epoch)
    features = pd.DataFrame(
        {
            "loan_date": loan_date.astype("int64"),
            "amount": pd.to_numeric(work["amount"], errors="coerce"),
            "duration": pd.to_numeric(work["duration"], errors="coerce"),
            "payments": pd.to_numeric(work["payments"], errors="coerce"),
            "status": fill_categorical(work["status"]),
        },
        index=work.index,
    )
    if features[["amount", "duration", "payments"]].isna().any().any():
        raise ValueError("Found non-numeric amount/duration/payments in loan.")
    out = pd.concat([_id_frame(work, TABLE_SPECS["loan"]["id_columns"]), features], axis=1)
    return out, epoch


def transform_order(raw: pd.DataFrame, date_epoch: str | None) -> tuple[pd.DataFrame, None]:
    """Fill order categoricals; keep account_to and amount numeric."""
    _ = date_epoch
    work = raw.copy()
    features = pd.DataFrame(
        {
            "bank_to": fill_categorical(work["bank_to"]),
            "account_to": pd.to_numeric(work["account_to"], errors="coerce"),
            "amount": pd.to_numeric(work["amount"], errors="coerce"),
            "k_symbol": fill_categorical(work["k_symbol"]),
        },
        index=work.index,
    )
    if features[["account_to", "amount"]].isna().any().any():
        raise ValueError("Found non-numeric account_to/amount in order.")
    out = pd.concat([_id_frame(work, TABLE_SPECS["order"]["id_columns"]), features], axis=1)
    return out, None


def transform_trans(raw: pd.DataFrame, date_epoch: str | None) -> tuple[pd.DataFrame, str]:
    """Keep transaction IDs; encode categoricals; convert date to day offsets."""
    work = raw.copy()
    for col in ["type", "operation", "k_symbol", "bank"]:
        work[col] = fill_categorical(work[col])
    work["account"] = pd.to_numeric(work["account"], errors="coerce").fillna(0).astype("int64")
    work["amount"] = pd.to_numeric(work["amount"], errors="coerce")
    work["balance"] = pd.to_numeric(work["balance"], errors="coerce")
    if work["amount"].isna().any() or work["balance"].isna().any():
        raise ValueError("Found non-numeric amount/balance in trans.")
    trans_date, epoch = to_day_offsets(work["date"], date_epoch)
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
        },
        index=work.index,
    )
    out = pd.concat([_id_frame(work, TABLE_SPECS["trans"]["id_columns"]), features], axis=1)
    return out, epoch


_TRANSFORMERS = {
    "account": transform_account,
    "card": transform_card,
    "client": transform_client,
    "disp": transform_disp,
    "district": transform_district,
    "loan": transform_loan,
    "order": transform_order,
    "trans": transform_trans,
}

# Feature columns in the same order as the ground-truth domain files (IDs omitted).
_FEATURE_ORDER_BY_TABLE = {
    "account": ["frequency", "account_date"],
    "card": ["card_type", "issued"],
    "client": ["year", "month", "client_date", "gender"],
    "disp": ["disp_type"],
    "district": ["A2", "A3"] + [f"A{i}" for i in range(4, 17)],
    "loan": ["loan_date", "amount", "duration", "payments", "status"],
    "order": ["bank_to", "account_to", "amount", "k_symbol"],
    "trans": [
        "trans_date",
        "trans_type",
        "operation",
        "amount",
        "balance",
        "k_symbol",
        "bank",
        "account",
    ],
}


def domain_for_table(table: str) -> dict[str, dict[str, str]]:
    """Build a type-only domain dict (IDs excluded)."""
    spec = TABLE_SPECS[table]
    discrete = set(spec["discrete_columns"])
    domain: dict[str, dict[str, str]] = {}
    for col in _FEATURE_ORDER_BY_TABLE[table]:
        domain[col] = {"type": "discrete" if col in discrete else "continuous"}
    return domain


def preprocess_table(
    table: str,
    raw: pd.DataFrame,
    date_epoch: str | None,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder], str | None]:
    """Transform, label-encode, and order columns for one table."""
    spec = TABLE_SPECS[table]
    features, epoch = _TRANSFORMERS[table](raw, date_epoch)
    encoded, encoders = label_encode(features, spec["discrete_columns"])
    output_cols = spec["id_columns"] + _FEATURE_ORDER_BY_TABLE[table]
    encoded = encoded[output_cols]
    return encoded, encoders, epoch


def preprocess_berka_all_tables(
    input_dir: Path | str,
    output_dir: Path | str | None = None,
    *,
    sep: str | None = None,
    date_epoch: str | None = None,
    trans_sample_size: int | None = None,
    seed: int = 42,
    save_artifacts: bool = True,
) -> dict[str, Any]:
    """Preprocess all eight Berka tables and write ClavaDDPM inputs.

    Parameters
    ----------
    input_dir:
        Directory containing raw ``{table}.csv`` or ``{table}.asc`` files.
    output_dir:
        Where to write processed CSVs and domain JSON. Defaults to
        ``implementations/tabular_data/multi_table/data/berka``.
    sep:
        Optional field separator override. Default: auto-detect.
    date_epoch:
        Optional fixed YYMMDD epoch for day offsets. Default: earliest date
        in each table's date column.
    trans_sample_size:
        If set, randomly subsample this many rows from ``trans`` only (other
        tables are unchanged). Foreign keys remain valid because ``account``
        is not subsampled. If the table is smaller than this size, it is kept
        in full.
    seed:
        Random seed used when subsampling ``trans``.
    save_artifacts:
        If True, also write ``preprocess_meta.json`` and
        ``label_encoders.pkl``.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir is not None else _DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    all_encoders: dict[str, dict[str, LabelEncoder]] = {}
    date_epochs: dict[str, str | None] = {}
    n_rows: dict[str, int] = {}

    for table in TABLE_NAMES:
        path = find_table_path(input_dir, table)
        print(f"Loading {table} from {path} ...")
        raw = load_raw_table(path, table, sep)
        print(f"  {len(raw):,} rows, columns={list(raw.columns)}")
        if table == "trans" and trans_sample_size is not None:
            raw = sample_dataframe(raw, trans_sample_size, seed)
            print(f"  Subsampled trans to {len(raw):,} rows (seed={seed}).")
        encoded, encoders, epoch = preprocess_table(table, raw, date_epoch)
        csv_path = output_dir / f"{table}.csv"
        encoded.to_csv(csv_path, index=False)
        save_json(output_dir / f"{table}_domain.json", domain_for_table(table))
        print(f"  Wrote {csv_path}")
        print(f"  Wrote {output_dir / f'{table}_domain.json'}")
        if encoders:
            print("  LabelEncoder classes:")
            for col, le in encoders.items():
                print(f"    {col}: {list(le.classes_)}")
        if epoch is not None:
            print(f"  Date epoch (YYMMDD): {epoch}")
        all_encoders[table] = encoders
        date_epochs[table] = epoch
        n_rows[table] = len(encoded)

    save_json(output_dir / "dataset_meta.json", DEFAULT_DATASET_META)
    print(f"Wrote {output_dir / 'dataset_meta.json'}")

    if save_artifacts:
        preprocess_meta = {
            "date_epochs_yymmdd": date_epochs,
            "n_rows": n_rows,
            "tables": {
                table: {
                    "id_columns": TABLE_SPECS[table]["id_columns"],
                    "discrete_columns": TABLE_SPECS[table]["discrete_columns"],
                    "continuous_columns": TABLE_SPECS[table]["continuous_columns"],
                    "output_columns": TABLE_SPECS[table]["id_columns"]
                    + _FEATURE_ORDER_BY_TABLE[table],
                    "label_classes": {
                        col: list(le.classes_) for col, le in all_encoders[table].items()
                    },
                }
                for table in TABLE_NAMES
            },
            "date_epoch": date_epoch,
            "trans_sample_size": trans_sample_size,
            "seed": seed,
        }
        save_json(output_dir / "preprocess_meta.json", preprocess_meta)
        encoders_path = output_dir / "label_encoders.pkl"
        with encoders_path.open("wb") as f:
            pickle.dump({"encoders": all_encoders, "date_epochs_yymmdd": date_epochs}, f)
        print(f"Wrote {output_dir / 'preprocess_meta.json'}")
        print(f"Wrote {encoders_path}")

    print("Done.")
    return {
        "output_dir": output_dir,
        "n_rows": n_rows,
        "date_epochs": date_epochs,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess all Berka tables for ClavaDDPM multi-table training.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=_DEFAULT_INPUT_DIR,
        help="Directory with raw {table}.csv or {table}.asc files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help="Directory for processed CSVs and domain JSON files.",
    )
    parser.add_argument(
        "--sep",
        default=None,
        help="Field separator. Default: auto-detect (';' for .asc).",
    )
    parser.add_argument(
        "--date-epoch",
        default=None,
        help="Optional fixed YYMMDD epoch for date day-offsets.",
    )
    parser.add_argument(
        "--trans-sample-size",
        type=int,
        default=None,
        help="If set, randomly subsample this many rows from trans only.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for trans subsampling.",
    )
    parser.add_argument(
        "--no-artifacts",
        action="store_true",
        help="Do not write preprocess_meta.json or label_encoders.pkl.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    preprocess_berka_all_tables(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        sep=args.sep,
        date_epoch=args.date_epoch,
        trans_sample_size=args.trans_sample_size,
        seed=args.seed,
        save_artifacts=not args.no_artifacts,
    )
