"""Utilities for working with serialized tabular training artifacts."""

import pickle
import json
from pathlib import Path
from typing import Any


def extract_data_from_pkl(file_path: str | Path) -> Any:
	"""Load and return the Python object stored in a pickle file.

	Pickle files can execute arbitrary code when loaded. Only use this function
	with files from a trusted source. Loading also reads the complete object
	into memory.
	"""
	with Path(file_path).open("rb") as file:
		return pickle.load(file)


def read_json_file(file_path: str | Path) -> Any:
	"""Read and return the JSON value stored in a file."""
	with Path(file_path).open(encoding="utf-8") as file:
		return json.load(file)
