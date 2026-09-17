from fastapi import APIRouter, HTTPException, status, BackgroundTasks

from backend.app import database
from backend.app.models.experiment import Experiment, ExperimentCreate, ExperimentUpdate
from backend.app.services.benchmark import run_benchmark
from backend.app.services.trainer import run_training


router = APIRouter(prefix="/api/experiments", tags=["experiments"])


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
            metrics=result["metrics"],
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


@router.get("", response_model=list[Experiment])
def get_experiments() -> list[Experiment]:
    return [Experiment.model_validate(item) for item in database.list_experiments()]


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


@router.post("/{experiment_id}/train")
def train_experiment(experiment_id: int) -> dict:
    experiment = database.get_experiment(experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    result = run_training(experiment["config"])
    database.update_experiment(experiment_id, status=result["status"], metrics=result["metrics"])
    return result


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