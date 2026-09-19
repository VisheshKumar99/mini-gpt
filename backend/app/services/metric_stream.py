import asyncio

from backend.app.services.websocket_manager import manager


def publish_metric(
    experiment_id: int,
    metric: dict,
):
    try:
        loop = asyncio.get_running_loop()

        loop.create_task(
            manager.broadcast(
                experiment_id,
                metric,
            )
        )

    except RuntimeError:
        pass