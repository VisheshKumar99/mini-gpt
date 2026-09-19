from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ExperimentCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str = ""
    config: dict[str, Any] = Field(default_factory=dict)


class Experiment(ExperimentCreate):
    id: int
    status: str = "created"
    created_at: datetime
    metrics: dict[str, float] = Field(default_factory=dict)
    has_model: bool = False

    @field_validator("metrics", mode="before")
    @classmethod
    def _coerce_metrics(cls, value: Any) -> dict[str, float]:
        # Older rows stored metrics as a list of per-step dicts. Coerce those
        # into a flat summary dict built from the final step so reads don't fail.
        if isinstance(value, list):
            last = value[-1] if value else {}
            summary: dict[str, float] = {}
            for key in ("train_loss", "val_loss", "tokens_per_sec"):
                item = last.get(key) if isinstance(last, dict) else None
                if item is not None:
                    summary[key] = float(item)
            if value:
                summary["steps"] = float(len(value))
            return summary
        return value or {}


class ExperimentUpdate(BaseModel):
    status: str | None = None
    metrics: dict[str, float] | None = None


class GenerateRequest(BaseModel):
    prompt: str = ""
    max_new_tokens: int = Field(default=200, ge=1, le=2000)
    temperature: float = Field(default=0.8, gt=0, le=5)
    top_k: int | None = Field(default=20, ge=1)


class GenerateResponse(BaseModel):
    experiment_id: int
    prompt: str
    completion: str
