from collections.abc import Iterable


def summarize_losses(losses: Iterable[float]) -> dict[str, float]:
    values = [float(loss) for loss in losses]
    if not values:
        return {}
    return {
        "initial_loss": values[0],
        "final_loss": values[-1],
        "best_loss": min(values),
        "average_loss": sum(values) / len(values),
    }
