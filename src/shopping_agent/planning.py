from __future__ import annotations

import json
import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from pydantic import ValidationError

from shopping_agent.agent_models import (
    AgentIntent,
    HardConstraints,
    PlannerDraft,
    ResearchTask,
    SessionSnapshot,
    ShoppingPlan,
)
from shopping_agent.catalog import SQLiteCatalog
from shopping_agent.llm import LLMCallError, StructuredLLM


@dataclass(frozen=True)
class PlanResult:
    plan: ShoppingPlan
    llm_calls: int
    cache_hits: int
    input_tokens: int
    output_tokens: int
    warning: str | None = None


class Planner:
    def __init__(
        self,
        catalog: SQLiteCatalog,
        llm: StructuredLLM,
        prompt_path: Path,
    ) -> None:
        self.catalog = catalog
        self.llm = llm
        self.prompt_path = prompt_path

    def plan(
        self,
        message: str,
        previous: SessionSnapshot | None,
        token_budget: int,
    ) -> PlanResult:
        if token_budget < 32:
            draft = rule_based_draft(message, previous)
            plan = self._expand(draft, message, previous, "rules")
            return PlanResult(
                plan=plan,
                llm_calls=0,
                cache_hits=0,
                input_tokens=0,
                output_tokens=0,
                warning="Исчерпан токен-бюджет сессии",
            )
        system_prompt = (self.prompt_path / "planner" / "v2.md").read_text(encoding="utf-8")
        user_input = self._planner_input(message, previous)
        try:
            result = self.llm.generate_structured(
                system_prompt=system_prompt,
                user_input=user_input,
                response_model=PlannerDraft,
                prompt_version="planner-v2",
                token_budget=token_budget,
            )
            draft = result.value
            assert isinstance(draft, PlannerDraft)
            mode = "cache" if result.cache_hit else "llm"
            plan = self._expand(draft, message, previous, mode)
            return PlanResult(
                plan=plan,
                llm_calls=0 if result.cache_hit else 1,
                cache_hits=int(result.cache_hit),
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
            )
        except (LLMCallError, OSError, ValidationError) as error:
            draft = rule_based_draft(message, previous)
            plan = self._expand(draft, message, previous, "rules")
            return PlanResult(plan, 1, 0, 0, 0, str(error))

    def _planner_input(self, message: str, previous: SessionSnapshot | None) -> str:
        payload: dict[str, object] = {"request": message}
        if previous:
            cart_products = []
            for item in previous.cart.items:
                product = self.catalog.get_product(item.product_id)
                cart_products.append(
                    {
                        "product_id": product.product_id,
                        "title": product.title,
                        "category": product.category,
                    }
                )
            payload["previous"] = {
                "request": previous.original_request,
                "budget": (
                    str(previous.constraints.budget)
                    if previous.constraints.budget is not None
                    else None
                ),
                "cart": cart_products,
                "clarification_count": previous.clarification_count,
            }
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    def _expand(
        self,
        draft: PlannerDraft,
        message: str,
        previous: SessionSnapshot | None,
        mode: str,
    ) -> ShoppingPlan:
        rule_draft = rule_based_draft(message, previous)
        previous_constraints = previous.constraints if previous else None
        budget = (
            rule_draft.budget
            if rule_draft.budget is not None
            else draft.budget
            if draft.budget is not None
            else (previous_constraints.budget if previous_constraints else None)
        )
        currency = draft.currency or (
            previous_constraints.currency if previous_constraints else "RUB"
        )
        message_mentions_rating = bool(re.search(r"рейтинг|оценк", message.casefold()))
        min_rating = previous_constraints.min_rating if previous_constraints else None
        if rule_draft.min_rating is not None:
            min_rating = rule_draft.min_rating
        elif message_mentions_rating and draft.min_rating is not None:
            min_rating = draft.min_rating
        excluded_terms = deduplicate(
            [
                *(previous_constraints.excluded_terms if previous_constraints else []),
                *rule_draft.excluded_terms,
                *[term for term in draft.excluded_terms if explicitly_mentioned(term, message)],
            ]
        )
        clarification_count = previous.clarification_count if previous else 0
        intent = rule_draft.intent if rule_draft.intent is AgentIntent.MODIFY_CART else draft.intent
        questions: list[str] = []
        if intent is AgentIntent.BUILD_CART and budget is None:
            questions = rule_draft.questions[: max(0, 2 - clarification_count)]
            if not questions:
                questions = ["Какой максимальный бюджет корзины?"]
        broad_request = "переезд" in message.casefold() or "квартир" in message.casefold()
        task_queries = (
            deduplicate([*rule_draft.tasks, *draft.tasks])[:6]
            if broad_request
            else (draft.tasks or rule_draft.tasks)
        )
        preferences = deduplicate([*rule_draft.preferences, *draft.preferences])
        tasks = [
            make_task(index, query, message, preferences, excluded_terms)
            for index, query in enumerate(task_queries[:8], start=1)
        ]
        target_product_id = resolve_target_product(
            self.catalog, previous, draft.target_terms or message
        )
        if intent is AgentIntent.MODIFY_CART and target_product_id is None:
            questions = ["Какой товар из корзины нужно заменить?"][
                : max(0, 2 - clarification_count)
            ]
        constraints = HardConstraints(
            budget=budget,
            currency=currency,
            min_rating=min_rating,
            excluded_terms=excluded_terms,
            must_have=preferences,
            deal_breakers=excluded_terms,
        )
        return ShoppingPlan(
            intent=intent,
            constraints=constraints,
            preferences=preferences,
            tasks=tasks,
            target_product_id=target_product_id,
            clarification_questions=questions,
            assumptions=[],
            planner_mode=mode,
        )


def rule_based_draft(message: str, previous: SessionSnapshot | None) -> PlannerDraft:
    effective_message = message
    if previous and not previous.cart.items and previous.constraints.budget is None:
        effective_message = f"{previous.original_request} {message}"
    normalized = effective_message.casefold()
    intent = (
        AgentIntent.MODIFY_CART
        if previous and re.search(r"\b(замени|заменить|поменяй|вместо)\b", normalized)
        else AgentIntent.BUILD_CART
    )
    budget = extract_budget(normalized)
    if budget is None and previous and previous.constraints.budget is None:
        standalone_amount = re.fullmatch(r"\s*(\d[\d\s]*)\s*", message)
        if standalone_amount:
            budget = Decimal(standalone_amount.group(1).replace(" ", ""))
    minimum_rating = extract_rating(normalized)
    excluded: list[str] = []
    for match in re.finditer(r"без\s+([а-яёa-z]+)", normalized):
        term = match.group(1)
        if term not in {"шерсти", "шерсть"}:
            excluded.append(normalize_material(term))
    preferences: list[str] = []
    if any(word in normalized for word in ("тихий", "тише", "потише", "шум")):
        preferences.append("низкий уровень шума")
    if "готовлю часто" in normalized or "часто готовлю" in normalized:
        preferences.append("подходит для частой готовки")
    if "тритан" in normalized:
        preferences.append("тритановая чаша")
    if "аллерг" in normalized or "шерст" in normalized:
        preferences.append("подходит человеку с аллергией на шерсть")

    tasks: list[str] = []
    broad_move = "переезд" in normalized or "квартир" in normalized
    keyword_tasks = [
        (("блендер",), "тихий блендер" if "тиш" in normalized else "блендер"),
        (("кастрю",), "набор кастрюль"),
        (("сковор",), "сковорода для частой готовки"),
        (("чайник",), "электрический чайник"),
        (("пылес", "аллерг", "шерст"), "пылесос HEPA с насадкой для шерсти"),
        (("бель", "текстил", "одеял"), "постельное бельё хлопок без шерсти"),
        (("сетев", "удлинитель"), "сетевой фильтр с защитой"),
    ]
    if broad_move:
        tasks.extend(
            [
                "тихий блендер для частой готовки",
                "набор кастрюль для ежедневной готовки",
                "сковорода для частой готовки",
                "пылесос HEPA с насадкой для шерсти",
                "постельное бельё хлопок без шерсти",
                "сетевой фильтр с защитой",
            ]
        )
    else:
        for keywords, query in keyword_tasks:
            if any(keyword in normalized for keyword in keywords):
                tasks.append(query)
    if intent is AgentIntent.MODIFY_CART and previous:
        target_id = resolve_target_product_from_text(previous, normalized)
        if target_id:
            target_title = next(
                item.product.title
                for item in previous.selected
                if item.product.product_id == target_id
            )
            qualifier = ""
            if "тиш" in normalized or "потише" in normalized:
                qualifier = "тихий "
            elif "мощн" in normalized:
                qualifier = "мощный "
            elif "дешев" in normalized:
                qualifier = "недорогой "
            tasks = [f"{qualifier}{target_title}"]
    if not tasks:
        tasks = [effective_message.strip()]
    questions = []
    previous_budget = previous.constraints.budget if previous else None
    if intent is AgentIntent.BUILD_CART and budget is None and previous_budget is None:
        questions.append("Какой максимальный бюджет корзины?")
    return PlannerDraft(
        intent=intent,
        budget=budget,
        currency="RUB",
        min_rating=minimum_rating,
        excluded_terms=deduplicate(excluded),
        preferences=preferences,
        tasks=deduplicate(tasks)[:6],
        target_terms=message if intent is AgentIntent.MODIFY_CART else None,
        questions=questions,
    )


def extract_budget(text: str) -> Decimal | None:
    patterns = [
        (
            r"(?:бюджет|до|не больше|в пределах|максимум)\s*"
            r"(\d[\d\s]*(?:[.,]\d+)?)\s*(к|тыс(?:яч[аиу]?)?)?"
        ),
        r"(\d[\d\s]*(?:[.,]\d+)?)\s*(руб(?:лей|ля|ль|\.)?)",
        r"(\d[\d\s]*(?:[.,]\d+)?)\s*(к|тыс(?:яч[аиу]?))(?:\b|\.)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        amount = Decimal(match.group(1).replace(" ", "").replace(",", "."))
        if match.group(2) and not match.group(2).startswith("руб"):
            amount *= 1000
        return amount
    return None


def extract_rating(text: str) -> float | None:
    match = re.search(r"(?:рейтинг\w*\s*(?:от)?|только)\s*(\d(?:[.,]\d)?)\s*\+?", text)
    return float(match.group(1).replace(",", ".")) if match else None


def normalize_material(term: str) -> str:
    return {"стекла": "стекло", "шерсти": "шерсть"}.get(term, term)


def explicitly_mentioned(term: str, message: str) -> bool:
    normalized_term = normalize_material(term.casefold())
    normalized_message = message.casefold()
    stem = normalized_term[: max(4, len(normalized_term) - 2)]
    return stem in normalized_message


def deduplicate(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value.strip() for value in values if value.strip()))


def make_task(
    index: int,
    query: str,
    message: str,
    preferences: list[str],
    global_excluded: list[str],
) -> ResearchTask:
    normalized = f"{query} {message}".casefold()
    category = None
    required: dict[str, str] = {}
    review_aspect = None
    excluded = list(global_excluded)
    if "пылес" in normalized:
        category = "Уборка"
        review_aspect = "аллергия"
        if "hepa" in normalized or "аллерг" in normalized or "шерст" in normalized:
            required = {"фильтр": "HEPA 13", "насадка для шерсти": "есть"}
        excluded = [term for term in excluded if term != "шерсть"]
    elif "блендер" in normalized:
        category = "Кухонная техника"
        review_aspect = "шум" if any(word in normalized for word in ("тих", "шум")) else "мощность"
        if any(word in normalized for word in ("тих", "потише")):
            required["режим шума"] = "тихий"
        if "тритан" in normalized:
            required["материал чаши"] = "тритан"
    elif "кастр" in normalized or "сковор" in normalized:
        category = "Посуда"
        review_aspect = "очистка"
    elif "чайник" in normalized:
        category = "Кухонная техника"
    elif any(word in normalized for word in ("бель", "одеял", "текстил")):
        category = "Текстиль"
        if "аллерг" in normalized or "шерст" in normalized:
            required = {"шерсть": "нет"}
        excluded = [term for term in excluded if term != "шерсть"]
    elif "сетев" in normalized or "удлинител" in normalized:
        category = "Электроника"
    reason = preferences[0] if preferences else f"Закрывает подзадачу {query}"
    return ResearchTask(
        task_id=f"task-{index}",
        query=query,
        reason=reason,
        category=category,
        required_attributes=required,
        excluded_terms=excluded,
        review_aspect=review_aspect,
        priority=max(1, 10 - index),
    )


def resolve_target_product(
    catalog: SQLiteCatalog, previous: SessionSnapshot | None, target_terms: str
) -> str | None:
    if not previous or not previous.cart.items:
        return None
    normalized = target_terms.casefold()
    for item in previous.cart.items:
        product = catalog.get_product(item.product_id)
        significant = [
            word for word in re.findall(r"[а-яёa-z]+", product.title.casefold()) if len(word) > 4
        ]
        if product.product_id.casefold() in normalized or any(
            word in normalized for word in significant
        ):
            return product.product_id
    if len(previous.cart.items) == 1:
        return previous.cart.items[0].product_id
    return None


def resolve_target_product_from_text(previous: SessionSnapshot, text: str) -> str | None:
    for selected in previous.selected:
        words = re.findall(r"[а-яёa-z]+", selected.product.title.casefold())
        if any(word in text for word in words if len(word) > 4):
            return selected.product.product_id
    return previous.cart.items[0].product_id if len(previous.cart.items) == 1 else None
