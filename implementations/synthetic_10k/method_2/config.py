"""Load the pipeline config and provide shared paths."""

import os
from pathlib import Path

import yaml


PROJECT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_DIR / "config.yaml"
CONFIG_LOCAL_PATH = PROJECT_DIR / "config.local.yaml"  # git-ignored overrides (your SEC identity)
RAW_DIR = PROJECT_DIR / "data" / "raw"
PARSED_DIR = PROJECT_DIR / "data" / "parsed"
COMPANYFACTS_DIR = PROJECT_DIR / "data" / "companyfacts"
LOG_DIR = PROJECT_DIR / "logs"
# Where steps 06-09 write and read fictional companies. Set SYNTHETIC_DIR to keep a new
# batch apart from the committed 20 companies, e.g. data/gemini20/method_2/synthetic.
SYNTHETIC_DIR = Path(os.environ.get("SYNTHETIC_DIR") or PROJECT_DIR / "data" / "synthetic").resolve()

PLACEHOLDER = "FILL_IN"


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Read config.yaml, then apply top-level overrides from config.local.yaml."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if CONFIG_LOCAL_PATH.exists():
        with open(CONFIG_LOCAL_PATH, encoding="utf-8") as f:
            cfg.update(yaml.safe_load(f) or {})
    return cfg


def get_companies(cfg: dict) -> list[str]:
    """Return the uppercase tickers for the run_profile, with duplicates removed."""
    profile = cfg["run_profile"]
    if profile not in cfg["companies"]:
        raise ValueError(f"run_profile '{profile}' not found under companies: {list(cfg['companies'])}")
    return list(dict.fromkeys(t.strip().upper() for t in cfg["companies"][profile]))


def get_sec_identity(cfg: dict) -> str:
    """Build the "Name email" User-Agent SEC requires. Exits if placeholders remain."""
    identity = cfg.get("sec_identity") or {}
    name = str(identity.get("name", "")).strip()
    email = str(identity.get("email", "")).strip()
    if not name or not email or PLACEHOLDER in name or PLACEHOLDER in email or "@" not in email:
        raise SystemExit(
            f"Set sec_identity.name and sec_identity.email in {CONFIG_LOCAL_PATH.name} before running. "
            "SEC requires a real name and contact email.",
        )
    return f"{name} {email}"
