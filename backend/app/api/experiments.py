from fastapi import APIRouter, HTTPException, status

from backend.app import database
from backend.app.models.experiment import Experiment, ExperimentCreate, ExperimentUpdate
from backend.app.services.benchmark import run_benchmark
from backend.app.services.trainer import run_training


router = APIRouter(prefix="/experiments", tags=["experiments"])


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


@router.get("/{experiment_id}", response_model=Experiment)
def get_experiment(experiment_id: int) -> Experiment:
    experiment = database.get_experiment(experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return Experiment.model_validate(experiment)


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
