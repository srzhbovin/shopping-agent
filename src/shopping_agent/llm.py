from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Protocol, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from shopping_agent.agent_store import AgentStore

StructuredModel = TypeVar("StructuredModel", bound=BaseModel)


class LLMCallError(RuntimeError):
    pass


@dataclass(frozen=True)
class LLMResult:
    value: BaseModel
    input_tokens: int
    output_tokens: int
    cache_hit: bool


class StructuredLLM(Protocol):
    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_input: str,
        response_model: type[StructuredModel],
        prompt_version: str,
        token_budget: int,
    ) -> LLMResult: ...


class UnavailableLLMClient:
    def generate_structured(self, **_: object) -> LLMResult:
        raise LLMCallError("Локальная модель отключена в автоматическом тесте")


class LMStudioClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_seconds: float,
        max_output_tokens: int,
        max_retries: int,
        store: AgentStore,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries
        self.store = store

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_input: str,
        response_model: type[StructuredModel],
        prompt_version: str,
        token_budget: int,
    ) -> LLMResult:
        cache_key = self._cache_key(system_prompt, user_input, response_model, prompt_version)
        cached = self.store.get_cached_llm(cache_key)
        if cached:
            response_json, _, _ = cached
            return LLMResult(
                value=response_model.model_validate_json(response_json),
                input_tokens=0,
                output_tokens=0,
                cache_hit=True,
            )

        output_limit = min(self.max_output_tokens, max(token_budget, 32))
        error: Exception | None = None
        current_input = user_input
        for attempt in range(self.max_retries + 1):
            try:
                payload = self._request(system_prompt, current_input, output_limit)
                content = "".join(
                    item.get("content", "")
                    for item in payload.get("output", [])
                    if item.get("type") == "message"
                )
                parsed_json = extract_json_object(content)
                normalized_json = keep_known_fields(parsed_json, response_model)
                value = response_model.model_validate_json(normalized_json)
                stats = payload.get("stats", {})
                input_tokens = int(stats.get("input_tokens", 0))
                output_tokens = int(stats.get("total_output_tokens", 0))
                self.store.cache_llm(
                    cache_key,
                    self.model,
                    prompt_version,
                    value.model_dump_json(),
                    input_tokens,
                    output_tokens,
                )
                return LLMResult(value, input_tokens, output_tokens, False)
            except (httpx.TimeoutException, ValueError, ValidationError) as caught:
                error = caught
                break
            except (httpx.HTTPError, KeyError) as caught:
                error = caught
                if attempt < self.max_retries:
                    retry_instruction = (
                        "\nПредыдущий ответ не прошёл проверку. Верни только один корректный JSON."
                    )
                    current_input = user_input + retry_instruction
        raise LLMCallError(f"Локальная модель не вернула корректный план: {error}")

    def _request(self, system_prompt: str, user_input: str, output_limit: int) -> dict:
        response = httpx.post(
            f"{self.base_url}/chat",
            json={
                "model": self.model,
                "system_prompt": system_prompt,
                "input": user_input,
                "reasoning": "off",
                "temperature": 0,
                "max_output_tokens": output_limit,
                "store": False,
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

    def _cache_key(
        self,
        system_prompt: str,
        user_input: str,
        response_model: type[BaseModel],
        prompt_version: str,
    ) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "system": system_prompt,
                "input": user_input,
                "schema": response_model.__name__,
                "version": prompt_version,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def extract_json_object(text: str) -> str:
    stripped = text.strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end < start:
        raise ValueError("В ответе модели нет объекта JSON")
    candidate = stripped[start : end + 1]
    json.loads(candidate)
    return candidate


def keep_known_fields(text: str, response_model: type[BaseModel]) -> str:
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("Ответ модели должен быть объектом JSON")
    known = {key: value for key, value in payload.items() if key in response_model.model_fields}
    for key in ("excluded_terms", "preferences", "tasks", "questions"):
        if key in known:
            known[key] = normalize_string_list(known[key])
    target = known.get("target_terms")
    if isinstance(target, list):
        values = normalize_string_list(target)
        known["target_terms"] = " ".join(values) or None
    for key in ("budget", "min_rating"):
        value = known.get(key)
        if isinstance(value, str):
            match = re.search(r"\d+(?:[.,]\d+)?", value.replace(" ", ""))
            known[key] = match.group(0).replace(",", ".") if match else None
    return json.dumps(known, ensure_ascii=False, separators=(",", ":"))


def normalize_string_list(value: object) -> list[str]:
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    normalized: list[str] = []
    for item in items:
        if isinstance(item, str):
            normalized.append(item)
        elif isinstance(item, dict):
            candidate = next(
                (
                    item.get(key)
                    for key in ("query", "title", "name", "product_id", "value")
                    if isinstance(item.get(key), str)
                ),
                None,
            )
            if candidate:
                normalized.append(candidate)
    return normalized
