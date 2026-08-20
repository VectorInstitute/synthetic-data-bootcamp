from midst_toolkit.common.logger import log
from logging import INFO

SEPARATOR = "-" * 80

def log_metrics(header: str, results: dict[str, float]) -> None:
    """
    Helper function to log metrics associated with the results dictionary in a structured fashion. The header
    is used to separate out different families of metrics in the output.

    Args:
        header: String to describe the set of metrics that will be logged.
        results: Dictionary of metric names (keys) and metric values (values) to be logged.
    """
    log(INFO, f"\n{header}\n{SEPARATOR}\n")
    for metric_name, metric_value in results.items():
        log(INFO, rf"Metric: {metric_name}\tScore: {metric_value}")
    log(INFO, f"{SEPARATOR}\n")