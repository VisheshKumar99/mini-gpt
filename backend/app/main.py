from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from backend.app import database
from backend.app.api.experiments import router as experiments_router
from backend.app.services.websocket_manager import manager

def create_app() -> FastAPI:
    app = FastAPI(title="Micro-GPT Experiments API", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(experiments_router)

    

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.websocket("/ws/experiments/{experiment_id}")
    async def experiment_websocket(
        websocket: WebSocket,
        experiment_id: int,
    ):
        experiment = database.get_experiment(experiment_id)

        if experiment is None:
            await websocket.close(code=1008)
            return

        await manager.connect(experiment_id, websocket)

        try:
            while True:
                await websocket.receive_text()

        except WebSocketDisconnect:
            manager.disconnect(experiment_id, websocket)

    return app


app = create_app()
database.initialize_database()
