from fastapi import APIRouter, HTTPException, status, BackgroundTasks, WebSocket, WebSocketDisconnect

from backend.app import database
from backend.app.models.experiment import (
    Experiment,
    ExperimentCreate,
    ExperimentUpdate,
    GenerateRequest,
    GenerateResponse,
)
from backend.app.services.benchmark import run_benchmark
from backend.app.services.trainer import run_training
from backend.app.services import generation

from backend.app.services.websocket_manager import manager


router = APIRouter(prefix="/api/experiments", tags=["experiments"])


def _summarize_metrics(result: dict) -> dict[str, float]:
    """Build a flat summary dict for the experiments.metrics column.

    The full per-step time series lives in the experiment_metrics table;
    here we only keep scalar summary values that match the Experiment model.
    """
    steps = result.get("metrics") or []
    last = steps[-1] if steps else {}

    summary: dict[str, float] = {}
    for key in ("train_loss", "val_loss", "tokens_per_sec"):
        value = last.get(key)
        if value is not None:
            summary[key] = float(value)

    if steps:
        summary["steps"] = float(len(steps))
    if result.get("training_time") is not None:
        summary["training_time"] = float(result["training_time"])
    if result.get("actual_parameters") is not None:
        summary["actual_parameters"] = float(result["actual_parameters"])

    return summary


def run_training_background(
    experiment_id: int,
    config: dict,
):
    try:
        database.update_experiment(
            experiment_id,
            status="running",
        )

        result = run_training(config)

        database.update_experiment(
            experiment_id,
            status=result["status"],
            metrics=_summarize_metrics(result),
        )

    except Exception as exc:

        print(
            f"Experiment {experiment_id} failed:",
            exc,
        )

        database.update_experiment(
            experiment_id,
            status="failed",
        )


@router.post("", response_model=Experiment, status_code=status.HTTP_201_CREATED)
def create_experiment(payload: ExperimentCreate) -> Experiment:
    experiment = database.create_experiment(
        name=payload.name,
        description=payload.description,
        config=payload.config,
    )
    return Experiment.model_validate(experiment)


def _to_experiment(item: dict) -> Experiment:
    """Validate a DB row and flag whether a trained checkpoint exists."""
    experiment = Experiment.model_validate(item)
    experiment.has_model = generation.has_checkpoint(experiment.id)
    return experiment


@router.get("", response_model=list[Experiment])
def get_experiments() -> list[Experiment]:
    return [_to_experiment(item) for item in database.list_experiments()]


@router.post("/{experiment_id}/train")
def train_experiment(
    experiment_id: int,
    background_tasks: BackgroundTasks,
) -> dict:

    experiment = database.get_experiment(experiment_id)

    if experiment is None:
        raise HTTPException(
            status_code=404,
            detail="Experiment not found",
        )

    database.update_experiment(
        experiment_id,
        status="queued",
    )

    config = {
        **experiment["config"],
        "experiment_id": experiment_id,
    }

    background_tasks.add_task(
        run_training_background,
        experiment_id,
        config,
    )

    return {
        "id": experiment_id,
        "status": "queued",
    }


@router.patch("/{experiment_id}", response_model=Experiment)
def update_experiment(experiment_id: int, payload: ExperimentUpdate) -> Experiment:
    experiment = database.update_experiment(experiment_id, **payload.model_dump())
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return Experiment.model_validate(experiment)


@router.post("/{experiment_id}/benchmark")
def benchmark_experiment(experiment_id: int) -> dict:
    experiment = database.get_experiment(experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return run_benchmark(experiment["config"])


@router.get("/{experiment_id}/metrics")
def get_experiment_metrics(experiment_id: int) -> list[dict]:
    experiment = database.get_experiment(experiment_id)

    if experiment is None:
        raise HTTPException(
            status_code=404,
            detail="Experiment not found",
        )

    return database.get_experiment_metrics(experiment_id)


@router.post("/{experiment_id}/generate", response_model=GenerateResponse)
def generate(experiment_id: int, payload: GenerateRequest) -> GenerateResponse:
    experiment = database.get_experiment(experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found")

    try:
        completion = generation.generate_text(
            experiment_id,
            prompt=payload.prompt,
            max_new_tokens=payload.max_new_tokens,
            temperature=payload.temperature,
            top_k=payload.top_k,
        )
    except generation.ModelNotTrainedError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    return GenerateResponse(
        experiment_id=experiment_id,
        prompt=payload.prompt,
        completion=completion,
    )


@router.get("/{experiment_id}", response_model=Experiment)
def get_experiment(experiment_id: int) -> Experiment:
    experiment = database.get_experiment(experiment_id)

    if experiment is None:
        raise HTTPException(
            status_code=404,
            detail="Experiment not found",
        )

    return _to_experiment(experiment)


@router.websocket("/{experiment_id}/ws")
async def experiment_websocket(
    websocket: WebSocket,
    experiment_id: int,
):
    experiment = database.get_experiment(
        experiment_id
    )

    if experiment is None:
        await websocket.close(code=1008)
        return

    await manager.connect(
        experiment_id,
        websocket,
    )

    try:
        while True:
            await websocket.receive_text()

    except WebSocketDisconnect:
        manager.disconnect(
            experiment_id,
            websocket,
        )