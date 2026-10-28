"""Build Phase 4 fictional-company specs from the calibration profile.

Input: data/calibration.json (run method_2.step05_make_calibration first) and
data/sp500_constituents.csv (sector, sub-industry, and HQ state per real ticker).
Output: data/synthetic/{fictional_id}/{fy}/spec.json, one fictional company per file,
one fiscal year each (see PLAN.md open question 1).

Each spec starts from one real company-year in calibration.json as a template (see
PLAN.md Phase 3): its sector, sub-industry, and HQ state carry over directly, and its
financial metrics are scaled and lightly jittered rather than resampled independently,
so the balance sheet still reconciles (assets = liabilities + equity, enforced by
construction and then checked) and the size/margin/return profile stays internally
consistent with a real company's rather than a random combination of percentiles. A
fictional name and ticker are generated and checked against every real S&P 500 name
and ticker, not just the 50 downloaded, so no fictional company can be mistaken for a
real one.

Run from the project directory:
    python -m method_2.step06_make_specs [--n 20] [--seed 42]
"""

import argparse
import csv
import json
import random
import re
import string

from method_2.config import PROJECT_DIR, RAW_DIR, SYNTHETIC_DIR
from method_2.utils import setup_logging, write_atomic


logger = setup_logging("make_specs")

CALIBRATION_PATH = RAW_DIR.parent / "calibration.json"
CONSTITUENTS_PATH = PROJECT_DIR / "data" / "sp500_constituents.csv"
N_COMPANIES = 20
SEED = 42

# A fictional company's size and ratios stay near its real template's, not identical
# to it (so it isn't just a renamed real company) and not far enough to lose the
# template's profile (so Phase 6's narrative still reads as the right kind of company).
SIZE_FACTOR_RANGE = (0.7, 1.4)
RATIO_JITTER_RANGE = (0.85, 1.15)

NAME_PREFIXES = [
    "Northbridge",
    "Clearwater",
    "Summit",
    "Harborview",
    "Redcanyon",
    "Silverlake",
    "Ironwood",
    "Meridian",
    "Fieldstone",
    "Brightpath",
    "Cedarfield",
    "Bluestone",
    "Westgate",
    "Oakridge",
    "Riverton",
    "Crestline",
    "Amberwood",
    "Stonegate",
    "Highmark",
    "Lakeshore",
]
NAME_CORES = [
    "Dynamics",
    "Industries",
    "Technologies",
    "Systems",
    "Partners",
    "Ventures",
    "Solutions",
    "Logistics",
    "Materials",
    "Energy",
    "Capital",
    "Analytics",
    "Networks",
    "Instruments",
    "Resources",
    "Sciences",
    "Robotics",
    "Components",
    "Innovations",
    "Enterprises",
]
NAME_SUFFIXES = ["Inc.", "Corp.", "Holdings, Inc.", "Group, Inc.", "Co."]

CYBERSECURITY_FRAMEWORKS = [
    "the NIST Cybersecurity Framework",
    "ISO/IEC 27001",
    "the NIST Cybersecurity Framework and ISO/IEC 27001",
    "an internal enterprise risk management framework aligned to NIST",
]
CYBERSECURITY_OVERSIGHT = [
    "the Audit Committee of the Board of Directors",
    "a dedicated Technology and Cybersecurity Committee of the Board",
    "the full Board of Directors, with support from the Audit Committee",
]
INCIDENT_TYPES = [
    "a contained phishing-related account compromise affecting a limited number of employee accounts",
    "unauthorized access to a third-party vendor's system that briefly exposed limited customer contact information",
    "a ransomware attempt on internal IT systems that was detected and contained before data was exfiltrated",
]
INCIDENT_PROBABILITY = 0.15


def load_company_years() -> list[dict]:
    """Return calibration.json's per-company-year records, to sample templates from."""
    data = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    return data["company_years"]


def load_real_companies() -> dict[str, dict]:
    """Return name, sector, sub_industry, and hq_state per real S&P 500 ticker."""
    companies = {}
    with open(CONSTITUENTS_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            hq = re.sub(r"\[\d+\]$", "", row["Headquarters Location"]).strip()
            hq_state = hq.rsplit(",", 1)[-1].strip() if "," in hq else hq
            companies[row["Symbol"].strip().upper()] = {
                "name": row["Security"].strip(),
                "sector": row["GICS Sector"].strip(),
                "sub_industry": row["GICS Sub-Industry"].strip(),
                "hq_state": hq_state,
            }
    return companies


def generate_name(rng: random.Random, used: set[str], real_names: set[str]) -> str:
    """Return a fictional company name not already used and not a real S&P 500 name."""
    for _ in range(200):
        name = f"{rng.choice(NAME_PREFIXES)} {rng.choice(NAME_CORES)} {rng.choice(NAME_SUFFIXES)}"
        if name not in used and name.lower() not in real_names:
            return name
    raise RuntimeError("could not generate a unique fictional company name")


def generate_ticker(rng: random.Random, used: set[str], real_tickers: set[str]) -> str:
    """Return a fictional ticker not already used and not a real S&P 500 ticker."""
    for _ in range(200):
        ticker = "".join(rng.choices(string.ascii_uppercase, k=rng.choice((3, 4))))
        if ticker not in used and ticker not in real_tickers:
            return ticker
    raise RuntimeError("could not generate a unique fictional ticker")


def _jitter(rng: random.Random, value: float, *, lo: float, hi: float) -> float:
    return max(lo, min(hi, value * rng.uniform(*RATIO_JITTER_RANGE)))


def build_financials(rng: random.Random, template: dict) -> dict:
    """Scale and jitter a template company-year, keeping assets = liabilities + equity.

    Assets are scaled by a random size factor; equity is a jittered share of assets
    (clamped to a plausible range); liabilities are assets minus that same rounded
    equity, so the identity holds exactly rather than approximately after independent
    rounding of each figure. Revenue is reconstructed from the template's own asset
    turnover (revenue / assets) rather than scaled independently, keeping it
    proportionate to the fictional company's size; net and operating income follow
    from the template's own margins. Any metric the template itself lacks (not every
    real company reports every tag; see PLAN.md Phase 3) is simply left out here too.
    """
    assets = round(template["total_assets"] * rng.uniform(*SIZE_FACTOR_RANGE))

    equity_to_assets = template.get("equity_to_assets", template["stockholders_equity"] / template["total_assets"])
    equity = round(assets * _jitter(rng, equity_to_assets, lo=0.02, hi=0.95))
    liabilities = assets - equity

    financials = {
        "total_assets": assets,
        "total_liabilities": liabilities,
        "stockholders_equity": equity,
    }

    if "revenues" in template:
        turnover = _jitter(rng, template["revenues"] / template["total_assets"], lo=0.05, hi=5.0)
        revenues = round(assets * turnover)
        financials["revenues"] = revenues

        if "net_margin" in template:
            financials["net_income"] = round(revenues * _jitter(rng, template["net_margin"], lo=-0.9, hi=0.9))
        if "operating_margin" in template:
            financials["operating_income"] = round(
                revenues * _jitter(rng, template["operating_margin"], lo=-0.9, hi=0.9),
            )

    return financials


def build_cybersecurity(rng: random.Random) -> dict:
    """Return the Item 1C facts a fictional company needs.

    Framework, oversight, and whether it disclosed an incident (most real filings
    don't; see INCIDENT_PROBABILITY).
    """
    had_incident = rng.random() < INCIDENT_PROBABILITY
    cyber = {
        "framework": rng.choice(CYBERSECURITY_FRAMEWORKS),
        "oversight": rng.choice(CYBERSECURITY_OVERSIGHT),
        "had_disclosed_incident": had_incident,
    }
    if had_incident:
        cyber["incident_summary"] = rng.choice(INCIDENT_TYPES)
    return cyber


def check_accounting_identity(financials: dict) -> bool:
    """Return True if assets equal liabilities plus equity (PLAN.md Phase 4's gate)."""
    assets = financials["total_assets"]
    liabilities = financials["total_liabilities"]
    equity = financials["stockholders_equity"]
    return assets == liabilities + equity


def build_spec(
    rng: random.Random,
    template: dict,
    *,
    real_companies: dict[str, dict],
    used_names: set[str],
    used_tickers: set[str],
    real_names: set[str],
    real_tickers: set[str],
) -> dict:
    """Build one fictional company's spec from one real company-year template."""
    for _attempt in range(5):
        financials = build_financials(rng, template)
        if check_accounting_identity(financials):
            break
    else:
        raise RuntimeError(f"could not build a balanced spec from template {template['ticker']}")

    name = generate_name(rng, used_names, real_names)
    fictional_id = generate_ticker(rng, used_tickers, real_tickers)
    used_names.add(name)
    used_tickers.add(fictional_id)

    real_info = real_companies.get(template["ticker"], {})
    return {
        "fictional_id": fictional_id,
        "fictional_name": name,
        "fiscal_year": template["fiscal_year"],
        "sector": real_info.get("sector", ""),
        "sub_industry": real_info.get("sub_industry", ""),
        "hq_state": real_info.get("hq_state", ""),
        "template": {"ticker": template["ticker"], "fiscal_year": template["fiscal_year"]},
        "financials": financials,
        "cybersecurity": build_cybersecurity(rng),
    }


def main() -> None:
    """Sample --n templates (default N_COMPANIES) and write a spec.json for each."""
    parser = argparse.ArgumentParser(description="Build fictional-company specs.")
    parser.add_argument("--n", type=int, default=N_COMPANIES, help="number of fictional companies")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    company_years = load_company_years()
    real_companies = load_real_companies()
    real_names = {info["name"].lower() for info in real_companies.values()}
    real_tickers = set(real_companies)

    templates = rng.sample(company_years, min(args.n, len(company_years)))
    used_names: set[str] = set()
    used_tickers: set[str] = set()

    for template in templates:
        spec = build_spec(
            rng,
            template,
            real_companies=real_companies,
            used_names=used_names,
            used_tickers=used_tickers,
            real_names=real_names,
            real_tickers=real_tickers,
        )
        out_dir = SYNTHETIC_DIR / spec["fictional_id"] / str(spec["fiscal_year"])
        out_dir.mkdir(parents=True, exist_ok=True)
        write_atomic(out_dir / "spec.json", json.dumps(spec, indent=2))
        logger.info(
            "%s (%s): template %s FY%d, assets $%.1fB, sector %s",
            spec["fictional_id"],
            spec["fictional_name"],
            spec["template"]["ticker"],
            spec["template"]["fiscal_year"],
            spec["financials"]["total_assets"] / 1e9,
            spec["sector"],
        )

    logger.info("Wrote %d specs to %s", len(templates), SYNTHETIC_DIR)


if __name__ == "__main__":
    main()
