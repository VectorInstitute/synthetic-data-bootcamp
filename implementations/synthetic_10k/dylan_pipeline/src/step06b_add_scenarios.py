"""Phase 4.5: add a narrative `scenario` to each fictional company's spec.json.

Input/output: data/synthetic/{fictional_id}/{fy}/spec.json (updated in place).

Why: spec.json originally held only size/sector/margin facts plus one cybersecurity
flag, so Phase 6's narrative had nothing distinguishing to say beyond generic
sector-template boilerplate -- which is also the root of Phase 7's "uniform prose"
finding (the same Liquidity paragraph and commitments sentence verbatim across all
20 companies). It also meant there was no structured ground truth for the
*qualitative* claims a RAG benchmark's equity-analysis questions would actually ask
about (an acquisition, a product launch, a regulatory event), only for the numbers.

This script assigns each company one scenario: a type, a one-paragraph summary, and
structured `facts` the narrative (step06c) must ground every scenario-specific claim
in. No number here is new: every dollar figure a scenario cites is copied from the
company's own, already-reconciled financials.json (see each entry's comment below),
so this never touches financials.json and the Phase 7 numeric gate's result doesn't
change. A deliberate choice, not an oversight: a scenario severe enough to need new
numbers (e.g. true going-concern distress) isn't attempted here, since forcing it
onto an already-reconciled balance sheet risks breaking what Phase 5 already
validated. "Elevated financial risk" scenarios (BIDZ, NDE, TVZ) instead reuse each
company's own already-thin margin or equity cushion as the grounding fact.

Run from the project directory:
    python -m src.step06b_add_scenarios
"""

import json
from pathlib import Path

from src.config import RAW_DIR
from src.utils import setup_logging, write_atomic


logger = setup_logging("add_scenarios")

SYNTHETIC_DIR = RAW_DIR.parent / "synthetic"

# Every dollar figure below is copied verbatim from that company's own financials.json
# (see PLAN.md's scenario-retrofit section for the exact field each one came from).
SCENARIOS = {
    "EASS": {
        "type": "mergers_and_acquisitions",
        "summary": (
            "Cedarfield Systems Holdings completed the acquisition of Thornridge Packaging Corp., "
            "a fictional packaging manufacturer, during fiscal 2025, consistent with its history of "
            "bolt-on acquisitions in the packaging industry."
        ),
        "facts": {
            "acquired_name": "Thornridge Packaging Corp.",
            "purchase_price": 249712064,  # == |other_investing_activities|
            "rationale": "expand manufacturing capacity and customer relationships in adjacent packaging end markets",
            "integration_status": "substantially integrated into existing operations as of fiscal year end",
        },
    },
    "FTC": {
        "type": "mergers_and_acquisitions",
        "summary": (
            "Oakridge Capital Holdings completed the acquisition of Carrowgate Capital Partners, a "
            "fictional investment management firm, during fiscal 2024, continuing its history of "
            "growth through acquiring complementary investment management and distribution businesses."
        ),
        "facts": {
            "acquired_name": "Carrowgate Capital Partners, Inc.",
            "purchase_price": 2211153641,  # == |other_investing_activities|
            "rationale": "add complementary investment strategies and expand third-party distribution reach",
            "integration_status": "being integrated into existing investment management and distribution operations",
        },
    },
    "PRG": {
        "type": "mergers_and_acquisitions",
        "summary": (
            "Amberwood Technologies completed the acquisition of Dunmore Technologies, a fictional "
            "technology-enabled instruments business, during fiscal 2024, consistent with its "
            "acquisitive growth strategy of acquiring and operating niche software and instrumentation businesses."
        ),
        "facts": {
            "acquired_name": "Dunmore Technologies, Inc.",
            "purchase_price": 2942974801,  # == |other_investing_activities| (== |cash_from_investing|, no separate capex)
            "rationale": "add a durable, high-margin niche instrumentation business to the existing portfolio",
            "integration_status": "operated as a standalone business within the existing decentralized operating model",
        },
    },
    "GOG": {
        "type": "product_launch",
        "summary": (
            "Riverton Partners launched SentinelFlow, a fictional new module extending its "
            "observability platform into automated anomaly detection, during fiscal 2023. The "
            "associated research and development investment is reflected in the year's operating loss."
        ),
        "facts": {
            "product_name": "SentinelFlow",
            "launch_timing": "introduced during fiscal 2023",
            "target_use_case": "automated anomaly detection across customers' multi-cloud infrastructure",
            "rationale": "extend the platform beyond metrics and log monitoring into automated, real-time anomaly detection",
        },
    },
    "HZP": {
        "type": "competitive_displacement",
        "summary": (
            "Stonegate Sciences experienced reduced usage from a significant customer during fiscal "
            "2025 after that customer migrated a portion of its workloads to a competing observability "
            "vendor, contributing to the year's thin operating margin."
        ),
        "facts": {
            "driver": "a significant customer migrated a portion of its workloads to a competing observability platform",
            "response": "continued investment in platform capabilities and customer success to limit further displacement",
        },
    },
    "BIDZ": {
        "type": "margin_crisis",
        "summary": (
            "Clearwater Technologies Group's fiscal 2025 results reflect severe and sustained "
            "reimbursement-rate pressure across its pharmacy, benefits, and care-delivery businesses, "
            "producing the thinnest margins in its reporting history (net margin of 0.4%)."
        ),
        "facts": {
            "driver": "continued reimbursement-rate reductions from government and commercial payors",
            "severity": "net margin of 0.4% and operating margin of 1.1%, both compressed to levels management considers unsustainable without corrective action",
        },
    },
    "NDE": {
        "type": "capital_structure",
        "summary": (
            "Harborview Logistics maintains a deliberately thin equity base (approximately 2.0% of "
            "total assets) funded by an ongoing debt-financed share repurchase program, a capital "
            "allocation strategy rather than a sign of distress, though it leaves the company with "
            "elevated financial leverage."
        ),
        "facts": {
            "driver": "an ongoing share repurchase program funded substantially through debt issuance",
            "framing": "a deliberate capital allocation choice reflecting confidence in cash flow generation, not financial distress",
            "risk_noted": "the resulting leverage increases vulnerability to adverse changes in interest rates or operating performance",
        },
    },
    "TVZ": {
        "type": "capital_structure",
        "summary": (
            "Meridian Innovations maintains a deliberately thin equity base (approximately 2.0% of "
            "total assets) funded by an ongoing debt-financed share repurchase program, a capital "
            "allocation strategy rather than a sign of distress, though it leaves the company with "
            "elevated financial leverage."
        ),
        "facts": {
            "driver": "an ongoing share repurchase program funded substantially through debt issuance",
            "framing": "a deliberate capital allocation choice reflecting confidence in cash flow generation, not financial distress",
            "risk_noted": "the resulting leverage increases vulnerability to adverse changes in interest rates or operating performance",
        },
    },
    "WJE": {
        "type": "cash_burn",
        "summary": (
            "Fieldstone Partners recorded an operating and net loss for fiscal 2025, reflecting "
            "ongoing investment in research and development and sales and marketing ahead of "
            "sustained profitability, consistent with its stage of growth as a subscription-based "
            "cybersecurity platform."
        ),
        "facts": {
            "driver": "ongoing investment in research and development and sales and marketing ahead of sustained profitability",
            "framing": "consistent with the company's growth stage as a subscription-based platform business, not a deteriorating core business",
        },
    },
    "BAI": {
        "type": "regulatory_rate_case",
        "summary": (
            "Northbridge Technologies' fiscal 2025 results reflect the outcome of a rate case in "
            "which state utility regulators approved a rate increase below the full amount the "
            "company requested, limiting full recovery of certain operating and capital costs."
        ),
        "facts": {
            "driver": "state utility regulators approved a rate increase below the full amount requested",
            "effect": "limited full recovery of certain operating and capital costs during the fiscal year",
        },
    },
    "VQWP": {
        "type": "weather_event",
        "summary": (
            "Westgate Technologies' fiscal 2023 capital spending reflects its ongoing multi-year "
            "infrastructure program together with incremental storm restoration and system-hardening "
            "work following significant weather events affecting its service territory during the year."
        ),
        "facts": {
            "driver": "significant weather events affecting the service territory during fiscal 2023",
            "effect": "incremental storm restoration and system-hardening work layered onto the ongoing capital program",
        },
    },
    "KMW": {
        "type": "regulatory_pricing_pressure",
        "summary": (
            "Northbridge Technologies Corp.'s fiscal 2024 margins reflect recently enacted state and "
            "federal measures affecting pharmacy benefit management practices and prescription drug "
            "pricing across its integrated pharmacy, benefits, and care-delivery businesses."
        ),
        "facts": {
            "driver": "recently enacted state and federal measures affecting pharmacy benefit management practices and prescription drug pricing",
            "effect": "contributed to a high effective tax rate and thin margins relative to the company's scale",
        },
    },
    "GKRH": {
        "type": "regulatory_consent_order",
        "summary": (
            "Cedarfield Materials Corp. entered into a consent order with a federal banking regulator "
            "during fiscal 2025 requiring enhancements to its risk management and compliance programs, "
            "increasing its compliance-related operating costs."
        ),
        "facts": {
            "driver": "a consent order with a federal banking regulator requiring risk management and compliance enhancements",
            "effect": "increased compliance-related operating costs reflected within other operating expenses",
        },
    },
    # Baseline: ordinary course of business, no material scenario event this fiscal year.
    # Real 10-Ks are mostly like this, so keeping a control group matters as much as the
    # scenario coverage does -- a benchmark where every company has a dramatic event would
    # itself be unrealistic.
    "CRJ": {"type": "baseline", "summary": "Ordinary course of business; no material scenario event during the fiscal year.", "facts": {}},
    "JFG": {"type": "baseline", "summary": "Ordinary course of business; no material scenario event during the fiscal year.", "facts": {}},
    "PKPM": {"type": "baseline", "summary": "Ordinary course of business; no material scenario event during the fiscal year.", "facts": {}},
    "SKR": {"type": "baseline", "summary": "Ordinary course of business; no material scenario event during the fiscal year.", "facts": {}},
    "TOUN": {"type": "baseline", "summary": "Ordinary course of business; no material scenario event during the fiscal year.", "facts": {}},
    "XTE": {"type": "baseline", "summary": "Ordinary course of business; no material scenario event during the fiscal year.", "facts": {}},
    "FAI": {"type": "baseline", "summary": "Ordinary course of business; no material scenario event during the fiscal year.", "facts": {}},
}


def main() -> None:
    """Add `scenario` to every spec.json under SYNTHETIC_DIR, keyed by fictional_id."""
    spec_paths = sorted(SYNTHETIC_DIR.glob("*/*/spec.json"))
    missing = set(SCENARIOS) - {json.loads(p.read_text(encoding="utf-8"))["fictional_id"] for p in spec_paths}
    if missing:
        raise SystemExit(f"SCENARIOS has entries for companies not found on disk: {sorted(missing)}")

    updated = 0
    for spec_path in spec_paths:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        fictional_id = spec["fictional_id"]
        if fictional_id not in SCENARIOS:
            raise SystemExit(f"No scenario assigned for {fictional_id}; add one to SCENARIOS before running")
        spec["scenario"] = SCENARIOS[fictional_id]
        write_atomic(spec_path, json.dumps(spec, indent=2))
        updated += 1
        logger.info("%s: scenario = %s", fictional_id, spec["scenario"]["type"])

    logger.info("Updated %d spec.json files with a scenario field", updated)


if __name__ == "__main__":
    main()
