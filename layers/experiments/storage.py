"""
实验存储 — SQLite 持久化
"""
import sqlite3
import json
from pathlib import Path
from datetime import datetime


class ExperimentStorage:
    """基于 SQLite 的实验数据存储"""

    def __init__(self, db_path: str = "experiments.db"):
        self.db_path = Path(db_path)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS experiments (
                    exp_id TEXT PRIMARY KEY,
                    name TEXT,
                    config TEXT,
                    config_hash TEXT,
                    tags TEXT,
                    status TEXT DEFAULT 'running',
                    created_at TEXT,
                    completed_at TEXT,
                    final_metrics TEXT
                );
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    exp_id TEXT,
                    step INTEGER,
                    metrics TEXT,
                    timestamp TEXT,
                    FOREIGN KEY (exp_id) REFERENCES experiments(exp_id)
                );
                CREATE TABLE IF NOT EXISTS artifacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    exp_id TEXT,
                    name TEXT,
                    path TEXT,
                    timestamp TEXT,
                    FOREIGN KEY (exp_id) REFERENCES experiments(exp_id)
                );
                CREATE INDEX IF NOT EXISTS idx_exp_status ON experiments(status);
                CREATE INDEX IF NOT EXISTS idx_metrics_exp ON metrics(exp_id);
            """)

    def save_experiment(self, data: dict) -> None:
        with self._get_conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO experiments
                   (exp_id, name, config, config_hash, tags, status, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (data["exp_id"], data["name"],
                 json.dumps(data.get("config", {})),
                 data.get("config_hash", ""),
                 json.dumps(data.get("tags", [])),
                 data.get("status", "running"),
                 data.get("created_at", datetime.now().isoformat())),
            )

    def save_metrics(self, exp_id: str, data: dict) -> None:
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO metrics (exp_id, step, metrics, timestamp) VALUES (?, ?, ?, ?)",
                (exp_id, data.get("step"),
                 json.dumps(data.get("metrics", {})),
                 data.get("timestamp", datetime.now().isoformat())),
            )

    def save_artifact(self, exp_id: str, data: dict) -> None:
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO artifacts (exp_id, name, path, timestamp) VALUES (?, ?, ?, ?)",
                (exp_id, data["name"], data["path"],
                 data.get("timestamp", datetime.now().isoformat())),
            )

    def update_experiment(self, exp_id: str, data: dict) -> None:
        updates = []
        values = []
        for k, v in data.items():
            if k in ("final_metrics",):
                v = json.dumps(v) if isinstance(v, dict) else v
            updates.append(f"{k} = ?")
            values.append(v)
        values.append(exp_id)

        with self._get_conn() as conn:
            conn.execute(
                f"UPDATE experiments SET {', '.join(updates)} WHERE exp_id = ?",
                values,
            )

    def query_experiments(self, status: str = None,
                          tag: str = None) -> list[dict]:
        query = "SELECT * FROM experiments WHERE 1=1"
        params = []
        if status:
            query += " AND status = ?"
            params.append(status)

        with self._get_conn() as conn:
            rows = conn.execute(query, params).fetchall()

        results = []
        for row in rows:
            d = dict(row)
            for key in ("config", "tags", "final_metrics"):
                if d.get(key) and isinstance(d[key], str):
                    try:
                        d[key] = json.loads(d[key])
                    except json.JSONDecodeError:
                        pass

            # Tag filter
            if tag and tag not in d.get("tags", []):
                continue

            results.append(d)
        return results

    def compare_experiments(self, exp_ids: list[str],
                            metrics: list[str] = None) -> list[dict]:
        rows = []
        for eid in exp_ids:
            with self._get_conn() as conn:
                row = conn.execute(
                    "SELECT * FROM experiments WHERE exp_id = ?", (eid,)
                ).fetchone()
                if row:
                    d = dict(row)
                    if d.get("final_metrics") and isinstance(d["final_metrics"], str):
                        d["final_metrics"] = json.loads(d["final_metrics"])
                    rows.append(d)
        return rows
