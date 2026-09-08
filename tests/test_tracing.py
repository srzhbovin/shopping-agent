import sys
from types import SimpleNamespace
from typing import Any

from shopping_agent.tracing import LangfuseBridge


class FakeObservation:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.updates: list[dict[str, Any]] = []

    def __enter__(self) -> "FakeObservation":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def update(self, **values: Any) -> None:
        self.updates.append(values)


class FakeLangfuse:
    instance: "FakeLangfuse"

    def __init__(self, **settings: Any) -> None:
        self.settings = settings
        self.observations: list[FakeObservation] = []
        self.flush_calls = 0
        FakeLangfuse.instance = self

    def start_as_current_observation(self, **payload: Any) -> FakeObservation:
        observation = FakeObservation(payload)
        self.observations.append(observation)
        return observation

    def flush(self) -> None:
        self.flush_calls += 1


def test_langfuse_bridge_records_agent_generation_and_usage(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "langfuse", SimpleNamespace(Langfuse=FakeLangfuse))
    bridge = LangfuseBridge("public", "secret", "http://langfuse.local")

    with bridge.trace(
        session_id="session-12345678",
        trace_id="trace-12345678",
        input_message="Подбери чайник",
    ) as root:
        bridge.update(root, {"status": "completed"})
        with bridge.span("planner", "generation", {"message": "Подбери чайник"}) as step:
            bridge.update_generation(
                step,
                output={"tasks": ["чайник"]},
                model="qwen/qwen3.5-4b",
                input_tokens=20,
                output_tokens=10,
            )
    bridge.flush()

    client = FakeLangfuse.instance
    assert bridge.enabled
    assert client.settings == {
        "public_key": "public",
        "secret_key": "secret",
        "host": "http://langfuse.local",
    }
    assert [item.payload["as_type"] for item in client.observations] == [
        "agent",
        "generation",
    ]
    assert client.observations[0].updates == [{"output": {"status": "completed"}}]
    assert client.observations[1].updates[0]["usage_details"] == {
        "input": 20,
        "output": 10,
        "total": 30,
    }
    assert client.flush_calls == 1


def test_langfuse_bridge_without_keys_is_noop() -> None:
    bridge = LangfuseBridge(None, None, "http://langfuse.local")

    with bridge.trace(
        session_id="session-12345678",
        trace_id="trace-12345678",
        input_message="Подбери чайник",
    ) as root:
        bridge.update(root, {"ignored": True})
    with bridge.span("planner", "generation", {}) as step:
        bridge.update_generation(
            step,
            output={},
            model="qwen/qwen3.5-4b",
            input_tokens=0,
            output_tokens=0,
        )
    bridge.flush()

    assert not bridge.enabled
