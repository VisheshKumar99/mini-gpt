from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ExperimentCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str = ""
    config: dict[str, Any] = Field(default_factory=dict)


class Experiment(ExperimentCreate):
    id: int
    status: str = "created"
    created_at: datetime
    metrics: dict[str, float] = Field(default_factory=dict)


class ExperimentUpdate(BaseModel):
    status: str | None = None
    metrics: dict[str, float] | None = None
