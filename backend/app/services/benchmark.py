from typing import Any


def run_benchmark(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the benchmark configuration and an empty result set."""
    return {
        "status": "not_run",
        "config": config or {},
        "results": {},
    }
