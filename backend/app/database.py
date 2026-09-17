import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATABASE_PATH = Path(
    os.getenv("MICRO_GPT_DATABASE", Path(__file__).resolve().parents[2] / "experiments.db")
)


def get_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database() -> None:
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                config TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL DEFAULT 'created',
                metrics TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            )
            """
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS experiment_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL,
                step INTEGER NOT NULL,
                train_loss REAL,
                val_loss REAL,
                tokens_per_sec REAL,
                tokens_seen INTEGER,
                elapsed_s REAL,
                created_at TEXT NOT NULL,

                FOREIGN KEY (experiment_id)
                    REFERENCES experiments(id)
                    ON DELETE CASCADE
            )
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_experiment_metrics_experiment
            ON experiment_metrics(experiment_id, step)
            """
        )


def row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "name": row["name"],
        "description": row["description"],
        "config": json.loads(row["config"]),
        "status": row["status"],
        "metrics": json.loads(row["metrics"]),
        "created_at": row["created_at"],
    }


def create_experiment(name: str, description: str, config: dict[str, Any]) -> dict[str, Any]:
    created_at = datetime.now(timezone.utc).isoformat()
    with get_connection() as connection:
        cursor = connection.execute(
            "INSERT INTO experiments (name, description, config, created_at) VALUES (?, ?, ?, ?)",
            (name, description, json.dumps(config), created_at),
        )
        row = connection.execute(
            "SELECT * FROM experiments WHERE id = ?", (cursor.lastrowid,)
        ).fetchone()
    return row_to_dict(row)


def list_experiments() -> list[dict[str, Any]]:
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT * FROM experiments ORDER BY created_at DESC"
        ).fetchall()
    return [row_to_dict(row) for row in rows]


def get_experiment(experiment_id: int) -> dict[str, Any] | None:
    with get_connection() as connection:
        row = connection.execute(
            "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
        ).fetchone()
    return row_to_dict(row) if row else None


def update_experiment(experiment_id: int, **changes: Any) -> dict[str, Any] | None:
    allowed = {key: value for key, value in changes.items() if value is not None}
    if not allowed:
        return get_experiment(experiment_id)

    serialized = {
        key: json.dumps(value) if key in {"config", "metrics"} else value
        for key, value in allowed.items()
    }
    assignments = ", ".join(f"{key} = ?" for key in serialized)
    with get_connection() as connection:
        connection.execute(
            f"UPDATE experiments SET {assignments} WHERE id = ?",
            (*serialized.values(), experiment_id),
        )
    return get_experiment(experiment_id)

def add_experiment_metric(
    experiment_id: int,
    metric: dict[str, Any],
) -> None:

    created_at = datetime.now(timezone.utc).isoformat()

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO experiment_metrics (
                experiment_id,
                step,
                train_loss,
                val_loss,
                tokens_per_sec,
                tokens_seen,
                elapsed_s,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                experiment_id,
                metric["step"],
                metric.get("train_loss"),
                metric.get("val_loss"),
                metric.get("tokens_per_sec"),
                metric.get("tokens_seen"),
                metric.get("elapsed_s"),
                created_at,
            ),
        )
def get_experiment_metrics(
    experiment_id: int,
) -> list[dict[str, Any]]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                step,
                train_loss,
                val_loss,
                tokens_per_sec,
                tokens_seen,
                elapsed_s
            FROM experiment_metrics
            WHERE experiment_id = ?
            ORDER BY step ASC
            """,
            (experiment_id,),
        ).fetchall()

    return [dict(row) for row in rows]
