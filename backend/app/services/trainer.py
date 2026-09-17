from typing import Any

from backend.app.services.metrics import summarize_losses


def run_training(config: dict[str, Any]) -> dict[str, Any]:
    """Return a training result placeholder for the existing CLI trainer.

    The actual PyTorch training loop remains in src/train.py. This service gives
    the API a stable result shape until training is moved behind the backend.
    """
    return {
        "status": "queued",
        "config": config,
        "metrics": summarize_losses([]),
    }
