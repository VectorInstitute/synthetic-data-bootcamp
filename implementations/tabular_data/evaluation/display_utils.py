"""Helpers for printing evaluation metrics in tabular notebooks."""

from logging import INFO

from midst_toolkit.common.logger import log


SEPARATOR = "-" * 80


def log_metrics(header: str, results: dict[str, float]) -> None:
    """Log metric names and values under a section header.

    The header is used to separate different families of metrics in the output.

    Args:
        header: String to describe the set of metrics that will be logged.
        results: Dictionary of metric names and values to be logged.
    """
    log(INFO, f"\n{header}\n{SEPARATOR}\n")
    for metric_name, metric_value in results.items():
        log(INFO, f"Metric: {metric_name}\tScore: {metric_value}")
    log(INFO, f"{SEPARATOR}\n")
