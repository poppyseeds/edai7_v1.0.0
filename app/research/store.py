"""Durable local experiment records; SQLite is sufficient for a single-machine project."""
import json
from pathlib import Path
import sqlite3
from datetime import datetime, timezone
from uuid import uuid4


def now():
    return datetime.now(timezone.utc).isoformat()


class ExperimentStore:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.database = self.root / "experiments.sqlite3"
        with self.connect() as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS experiments
                (id TEXT PRIMARY KEY, created TEXT NOT NULL, updated TEXT NOT NULL,
                 status TEXT NOT NULL, payload TEXT NOT NULL)""")

    def connect(self):
        connection = sqlite3.connect(self.database, timeout=30)
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def create(self, config: dict, dataset_hash: str, dataset_identifier: str):
        identifier = uuid4().hex
        directory = self.root / identifier
        directory.mkdir()
        payload = {"id": identifier, "status": "queued", "stage": "queued", "created": now(),
                   "config": config, "dataset_hash": dataset_hash, "dataset_identifier": dataset_identifier,
                   "events": [], "metrics": None, "failure": None}
        (directory / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        with self.connect() as connection:
            connection.execute("INSERT INTO experiments VALUES (?,?,?,?,?)",
                               (identifier, now(), now(), "queued", json.dumps(payload)))
        return identifier

    def get(self, identifier: str):
        with self.connect() as connection:
            row = connection.execute("SELECT payload FROM experiments WHERE id=?", (identifier,)).fetchone()
        if row is None:
            raise KeyError(identifier)
        return json.loads(row[0])

    def update(self, identifier: str, **changes):
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT payload FROM experiments WHERE id=?", (identifier,)).fetchone()
            if row is None:
                raise KeyError(identifier)
            payload = json.loads(row[0])
            payload.update(changes)
            payload["updated"] = now()
            connection.execute("UPDATE experiments SET status=?,updated=?,payload=? WHERE id=?",
                               (payload["status"], now(), json.dumps(payload, allow_nan=False), identifier))
        return payload

    def event(self, identifier: str, stage: str, message: str):
        record = self.get(identifier)
        self.update(identifier, stage=stage, events=record["events"] + [{"at": now(), "stage": stage, "message": message}])

    def list(self, limit: int = 100):
        with self.connect() as connection:
            rows = connection.execute("SELECT payload FROM experiments ORDER BY created DESC LIMIT ?",
                                      (min(max(limit, 1), 500),)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def recover_interrupted(self):
        with self.connect() as connection:
            identifiers = connection.execute("SELECT id FROM experiments WHERE status IN ('queued','running')").fetchall()
        for (identifier,) in identifiers:
            self.update(identifier, status="failed", stage="interrupted",
                        failure={"type": "Interrupted", "message": "Worker process stopped before completion; rerun explicitly."})

    def directory(self, identifier: str):
        self.get(identifier)  # require a real database record before using an identifier as a path
        return self.root / identifier
