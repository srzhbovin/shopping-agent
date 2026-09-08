from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from shopping_agent.agent_models import SessionSnapshot, TraceRecord


class AgentStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS agent_sessions (
                    session_id TEXT PRIMARY KEY,
                    snapshot_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS llm_cache (
                    cache_key TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS agent_traces (
                    trace_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    trace_json TEXT NOT NULL,
                    started_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_agent_traces_session
                ON agent_traces(session_id, started_at);
                """
            )

    def load_session(self, session_id: str) -> SessionSnapshot | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT snapshot_json FROM agent_sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        return SessionSnapshot.model_validate_json(row[0]) if row else None

    def save_session(self, snapshot: SessionSnapshot) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO agent_sessions(session_id, snapshot_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                  snapshot_json=excluded.snapshot_json,
                  updated_at=excluded.updated_at
                """,
                (snapshot.session_id, snapshot.model_dump_json(), snapshot.updated_at.isoformat()),
            )

    def get_cached_llm(self, cache_key: str) -> tuple[str, int, int] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT response_json, input_tokens, output_tokens
                FROM llm_cache WHERE cache_key = ?
                """,
                (cache_key,),
            ).fetchone()
        return (row[0], int(row[1]), int(row[2])) if row else None

    def cache_llm(
        self,
        cache_key: str,
        model: str,
        prompt_version: str,
        response_json: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO llm_cache(
                  cache_key, model, prompt_version, response_json, input_tokens, output_tokens
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    cache_key,
                    model,
                    prompt_version,
                    response_json,
                    input_tokens,
                    output_tokens,
                ),
            )

    def save_trace(self, trace: TraceRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO agent_traces(
                  trace_id, session_id, status, trace_json, started_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    trace.trace_id,
                    trace.session_id,
                    trace.status,
                    trace.model_dump_json(),
                    trace.started_at.isoformat(),
                ),
            )

    def load_trace(self, trace_id: str) -> TraceRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT trace_json FROM agent_traces WHERE trace_id = ?", (trace_id,)
            ).fetchone()
        return TraceRecord.model_validate_json(row[0]) if row else None
