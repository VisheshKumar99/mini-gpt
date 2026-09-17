from fastapi import FastAPI

from backend.app import database
from backend.app.api.experiments import router as experiments_router


def create_app() -> FastAPI:
    app = FastAPI(title="Micro-GPT Experiments API", version="0.1.0")
    app.include_router(experiments_router)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
database.initialize_database()
