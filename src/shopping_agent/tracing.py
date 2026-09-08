from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from typing import Any


class LangfuseBridge:
    def __init__(self, public_key: str | None, secret_key: str | None, host: str) -> None:
        self.enabled = bool(public_key and secret_key)
        self.client: Any | None = None
        if not self.enabled:
            return
        try:
            from langfuse import Langfuse

            self.client = Langfuse(public_key=public_key, secret_key=secret_key, host=host)
        except Exception:
            self.enabled = False
            self.client = None

    @contextmanager
    def trace(self, *, session_id: str, trace_id: str, input_message: str) -> Iterator[Any]:
        if not self.client:
            with nullcontext() as empty:
                yield empty
            return
        with self.client.start_as_current_observation(
            as_type="agent",
            name="shopping-agent",
            input={"session_id": session_id, "message": input_message},
            metadata={"local_trace_id": trace_id, "session_id": session_id},
        ) as observation:
            yield observation

    @contextmanager
    def span(self, name: str, as_type: str, input_data: Any) -> Iterator[Any]:
        if not self.client:
            with nullcontext() as empty:
                yield empty
            return
        with self.client.start_as_current_observation(
            as_type=as_type, name=name, input=input_data
        ) as observation:
            yield observation

    @staticmethod
    def update(observation: Any, output: Any) -> None:
        if observation is not None:
            observation.update(output=output)

    @staticmethod
    def update_generation(
        observation: Any,
        *,
        output: Any,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        if observation is not None:
            observation.update(
                output=output,
                model=model,
                usage_details={
                    "input": input_tokens,
                    "output": output_tokens,
                    "total": input_tokens + output_tokens,
                },
            )

    def flush(self) -> None:
        if self.client:
            self.client.flush()
