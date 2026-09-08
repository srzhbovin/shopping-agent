import json
from pathlib import Path

import pytest

from shopping_agent.agent_models import PlannerDraft
from shopping_agent.agent_store import AgentStore
from shopping_agent.llm import LLMCallError, LMStudioClient, extract_json_object


class FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {
            "output": [{"type": "message", "content": self.content}],
            "stats": {"input_tokens": 12, "total_output_tokens": 8},
        }


def test_lm_studio_client_validates_and_caches_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0
    content = json.dumps(
        {
            "intent": "build_cart",
            "budget": 8000,
            "currency": "RUB",
            "min_rating": 4,
            "excluded_terms": [],
            "preferences": ["тихий"],
            "tasks": ["тихий блендер"],
            "target_terms": [{"title": "блендер"}],
            "questions": [],
            "budget_confirmed": True,
        },
        ensure_ascii=False,
    )

    def fake_post(*args, **kwargs):
        nonlocal calls
        calls += 1
        return FakeResponse(content)

    monkeypatch.setattr("shopping_agent.llm.httpx.post", fake_post)
    client = LMStudioClient(
        base_url="http://lm.test/api/v1",
        model="qwen-test",
        timeout_seconds=1,
        max_output_tokens=128,
        max_retries=0,
        store=AgentStore(tmp_path / "agent.db"),
    )

    first = client.generate_structured(
        system_prompt="prompt",
        user_input="input",
        response_model=PlannerDraft,
        prompt_version="v1",
        token_budget=100,
    )
    second = client.generate_structured(
        system_prompt="prompt",
        user_input="input",
        response_model=PlannerDraft,
        prompt_version="v1",
        token_budget=100,
    )

    assert first.value.budget == 8000
    assert first.value.target_terms == "блендер"
    assert first.input_tokens == 12 and first.output_tokens == 8
    assert not first.cache_hit
    assert second.cache_hit
    assert second.input_tokens == 0 and second.output_tokens == 0
    assert calls == 1


def test_lm_studio_client_rejects_non_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "shopping_agent.llm.httpx.post", lambda *args, **kwargs: FakeResponse("не json")
    )
    client = LMStudioClient(
        base_url="http://lm.test/api/v1",
        model="qwen-test",
        timeout_seconds=1,
        max_output_tokens=64,
        max_retries=0,
        store=AgentStore(tmp_path / "agent.db"),
    )

    with pytest.raises(LLMCallError):
        client.generate_structured(
            system_prompt="prompt",
            user_input="input",
            response_model=PlannerDraft,
            prompt_version="v1",
            token_budget=64,
        )


def test_extract_json_ignores_surrounding_text() -> None:
    assert extract_json_object('ответ ```json\n{"value": 1}\n```') == '{"value": 1}'
