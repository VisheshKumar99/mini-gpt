import asyncio

from fastapi import WebSocket


class ConnectionManager:

    def __init__(self):
        self.connections: dict[
            int,
            list[WebSocket]
        ] = {}

        self.loop: asyncio.AbstractEventLoop | None = None

    async def connect(
        self,
        experiment_id: int,
        websocket: WebSocket,
    ):
        await websocket.accept()

        self.loop = asyncio.get_running_loop()

        self.connections.setdefault(
            experiment_id,
            [],
        ).append(websocket)

    def disconnect(
        self,
        experiment_id: int,
        websocket: WebSocket,
    ):
        connections = self.connections.get(
            experiment_id,
            [],
        )

        if websocket in connections:
            connections.remove(websocket)

        if not connections:
            self.connections.pop(
                experiment_id,
                None,
            )

    async def broadcast(
        self,
        experiment_id: int,
        metric: dict,
    ):
        connections = list(
            self.connections.get(
                experiment_id,
                [],
            )
        )

        for websocket in connections:
            try:
                await websocket.send_json(metric)
            except Exception:
                self.disconnect(
                    experiment_id,
                    websocket,
                )

    def publish(
        self,
        experiment_id: int,
        metric: dict,
    ):
        if self.loop is None:
            return

        asyncio.run_coroutine_threadsafe(
            self.broadcast(
                experiment_id,
                metric,
            ),
            self.loop,
        )


manager = ConnectionManager()