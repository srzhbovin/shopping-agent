from __future__ import annotations

import re
import time
from collections.abc import Callable
from typing import Any, Literal, TypedDict, TypeVar

from langgraph.graph import END, START, StateGraph

from shopping_agent.agent_models import (
    AgentEvent,
    AgentIntent,
    AgentRequest,
    AgentStatus,
    CritiqueIssue,
    CritiqueResult,
    RejectedAlternative,
    SelectedItem,
    SessionSnapshot,
    ShoppingPlan,
    ToolCallRecord,
)
from shopping_agent.catalog import SQLiteCatalog, product_matches_filters
from shopping_agent.config import Settings
from shopping_agent.errors import BudgetExceededError
from shopping_agent.models import (
    CartAction,
    CartOpsInput,
    CartState,
    CompareProductsInput,
    SearchCatalogInput,
    SearchFilters,
    SummarizeReviewsInput,
)
from shopping_agent.planning import Planner
from shopping_agent.tools import ShoppingTools
from shopping_agent.tracing import LangfuseBridge

ToolResult = TypeVar("ToolResult")


class AgentGraphState(TypedDict):
    request: AgentRequest
    session_id: str
    trace_id: str
    previous: SessionSnapshot | None
    plan: ShoppingPlan | None
    base_cart: CartState
    base_selected: list[SelectedItem]
    cart: CartState
    selected: list[SelectedItem]
    rejected: list[RejectedAlternative]
    critique: CritiqueResult | None
    tool_records: list[ToolCallRecord]
    events: list[AgentEvent]
    research_gaps: list[CritiqueIssue]
    blocked_product_ids: list[str]
    step_count: int
    llm_calls: int
    llm_cache_hits: int
    input_tokens: int
    output_tokens: int
    session_tokens_used: int
    critic_revisions: int
    retry_research: bool
    status: AgentStatus
    answer: str
    degraded_reasons: list[str]
    deadline_at: float
    stop_requested: bool


class ShoppingAgentGraph:
    def __init__(
        self,
        *,
        planner: Planner,
        tools: ShoppingTools,
        catalog: SQLiteCatalog,
        settings: Settings,
        observability: LangfuseBridge,
    ) -> None:
        self.planner = planner
        self.tools = tools
        self.catalog = catalog
        self.settings = settings
        self.observability = observability
        builder = StateGraph(AgentGraphState)
        builder.add_node("planner", self._planner)
        builder.add_node("researcher", self._researcher)
        builder.add_node("critic", self._critic)
        builder.add_node("finalizer", self._finalizer)
        builder.add_edge(START, "planner")
        builder.add_conditional_edges(
            "planner",
            self._after_planner,
            {"researcher": "researcher", "finalizer": "finalizer"},
        )
        builder.add_edge("researcher", "critic")
        builder.add_conditional_edges(
            "critic",
            self._after_critic,
            {"researcher": "researcher", "finalizer": "finalizer"},
        )
        builder.add_edge("finalizer", END)
        self.graph = builder.compile()

    def _planner(self, state: AgentGraphState) -> dict[str, Any]:
        event = AgentEvent(
            event="node_started",
            node="planner",
            message="Планировщик выделяет ограничения и подзадачи",
        )
        remaining_tokens = max(
            0,
            self.settings.agent_token_budget - state["session_tokens_used"],
        )
        with self.observability.span(
            "planner", "generation", {"message": state["request"].message}
        ) as span:
            result = self.planner.plan(
                state["request"].message,
                state["previous"],
                remaining_tokens,
            )
            self.observability.update_generation(
                span,
                output=result.plan.model_dump(mode="json"),
                model=self.settings.llm_model,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
            )
        events = [*state["events"], event]
        events.append(
            AgentEvent(
                event="node_finished",
                node="planner",
                message=(
                    f"Сформировано подзадач {len(result.plan.tasks)}"
                    if not result.plan.clarification_questions
                    else "Нужно уточнение пользователя"
                ),
                data={
                    "mode": result.plan.planner_mode,
                    "questions": result.plan.clarification_questions,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "cache_hit": bool(result.cache_hits),
                },
            )
        )
        degraded = list(state["degraded_reasons"])
        if result.warning:
            degraded.append(result.warning)
        stop_requested = time.time() >= state["deadline_at"]
        if stop_requested:
            degraded.append("Исчерпан лимит времени сессии")
        return {
            "plan": result.plan,
            "events": events,
            "step_count": state["step_count"] + 1,
            "llm_calls": state["llm_calls"] + result.llm_calls,
            "llm_cache_hits": state["llm_cache_hits"] + result.cache_hits,
            "input_tokens": state["input_tokens"] + result.input_tokens,
            "output_tokens": state["output_tokens"] + result.output_tokens,
            "session_tokens_used": (
                state["session_tokens_used"] + result.input_tokens + result.output_tokens
            ),
            "degraded_reasons": degraded,
            "stop_requested": stop_requested,
        }

    @staticmethod
    def _after_planner(state: AgentGraphState) -> Literal["researcher", "finalizer"]:
        assert state["plan"] is not None
        return (
            "finalizer"
            if state["plan"].clarification_questions or state["stop_requested"]
            else "researcher"
        )

    def _researcher(self, state: AgentGraphState) -> dict[str, Any]:
        plan = state["plan"]
        assert plan is not None and plan.constraints.budget is not None
        events = [
            *state["events"],
            AgentEvent(
                event="node_started",
                node="researcher",
                message="Исследователь ищет и проверяет товары",
                data={"revision": state["critic_revisions"]},
            ),
        ]
        records = list(state["tool_records"])
        rejected: list[RejectedAlternative] = []
        gaps: list[CritiqueIssue] = []
        blocked = set(state["blocked_product_ids"])
        if plan.intent is AgentIntent.BUILD_CART:
            cart = CartState(
                budget=plan.constraints.budget,
                currency=plan.constraints.currency,
                remaining=plan.constraints.budget,
            )
            selected: list[SelectedItem] = []
        else:
            cart = state["base_cart"]
            selected = [
                item
                for item in state["base_selected"]
                if item.product.product_id != plan.target_product_id
            ]

        for task in sorted(plan.tasks, key=lambda item: -item.priority):
            if time.time() >= state["deadline_at"]:
                gaps.append(
                    CritiqueIssue(
                        code="time_limit",
                        message="Исчерпан лимит времени сессии",
                        severity="error",
                        task_id=task.task_id,
                    )
                )
                break
            if len(records) >= self.settings.agent_max_tool_calls:
                gaps.append(
                    CritiqueIssue(
                        code="tool_limit",
                        message="Достигнут предел вызовов инструментов",
                        severity="error",
                        task_id=task.task_id,
                    )
                )
                break
            max_price = cart.remaining
            if plan.intent is AgentIntent.MODIFY_CART and plan.target_product_id:
                old_product = self.catalog.get_product(plan.target_product_id)
                max_price += old_product.price
            filters = SearchFilters(
                max_price=max_price,
                categories=[task.category] if task.category else [],
                min_rating=plan.constraints.min_rating,
                required_attributes=task.required_attributes,
                excluded_terms=task.excluded_terms,
                currency=plan.constraints.currency,
            )
            search_request = SearchCatalogInput(query=task.query, filters=filters, top_k=5)
            search = self._tool_call(
                records,
                "search_catalog",
                task.task_id,
                lambda request=search_request: self.tools.search_catalog(request),
                lambda result: [hit.product.product_id for hit in result.hits],
            )
            if search is not None and not search.hits and task.category:
                relaxed = filters.model_copy(update={"categories": []})
                relaxed_request = SearchCatalogInput(query=task.query, filters=relaxed, top_k=5)
                search = self._tool_call(
                    records,
                    "search_catalog",
                    task.task_id,
                    lambda request=relaxed_request: self.tools.search_catalog(request),
                    lambda result: [hit.product.product_id for hit in result.hits],
                )
            hits = (
                []
                if search is None
                else [
                    hit
                    for hit in search.hits
                    if hit.product.product_id not in blocked
                    and hit.product.product_id != plan.target_product_id
                    and hit.product.product_id not in {item.product.product_id for item in selected}
                    and candidate_matches_task(task.query, hit.product)
                ]
            )
            if plan.intent is AgentIntent.MODIFY_CART and plan.target_product_id:
                old_product = self.catalog.get_product(plan.target_product_id)
                hits = rank_modification_hits(
                    state["request"].message,
                    old_product,
                    hits,
                )
            if not hits:
                gaps.append(
                    CritiqueIssue(
                        code="no_result",
                        message=f"Для подзадачи «{task.query}» подходящий товар не найден",
                        severity="error",
                        task_id=task.task_id,
                    )
                )
                continue
            if len(hits) >= 2:
                comparison_request = CompareProductsInput(
                    product_ids=[hit.product.product_id for hit in hits[:3]],
                    criteria=task.criteria,
                )
                self._tool_call(
                    records,
                    "compare_products",
                    task.task_id,
                    lambda request=comparison_request: self.tools.compare_products(request),
                    lambda result: [row.product_id for row in result.rows],
                )
            chosen = None
            for hit in hits:
                product = self._tool_call(
                    records,
                    "get_product_details",
                    task.task_id,
                    lambda product_id=hit.product.product_id: self.tools.get_product_details(
                        product_id
                    ),
                    lambda result: [result.product_id],
                )
                if product is None:
                    continue
                try:
                    operation = CartOpsInput(
                        cart=cart,
                        action=(
                            CartAction.REPLACE
                            if plan.intent is AgentIntent.MODIFY_CART
                            else CartAction.ADD
                        ),
                        product_id=(
                            product.product_id if plan.intent is AgentIntent.BUILD_CART else None
                        ),
                        old_product_id=(
                            plan.target_product_id
                            if plan.intent is AgentIntent.MODIFY_CART
                            else None
                        ),
                        new_product_id=(
                            product.product_id if plan.intent is AgentIntent.MODIFY_CART else None
                        ),
                    )
                    cart_result = self._tool_call(
                        records,
                        "cart_ops",
                        task.task_id,
                        lambda operation=operation: self.tools.cart_ops(operation),
                        lambda result: [item.product_id for item in result.cart.items],
                    )
                except BudgetExceededError:
                    cart_result = None
                if cart_result is None:
                    continue
                cart = cart_result.cart
                review_request = SummarizeReviewsInput(
                    product_id=product.product_id,
                    aspect=task.review_aspect,
                )
                review_summary = self._tool_call(
                    records,
                    "summarize_reviews",
                    task.task_id,
                    lambda request=review_request: self.tools.summarize_reviews(request),
                    lambda result: [result.product_id],
                )
                reason = (
                    f"{task.reason}. Цена {product.price} {product.currency}, "
                    f"рейтинг {product.rating if product.rating is not None else 'не указан'}."
                )
                chosen = SelectedItem(
                    task_id=task.task_id,
                    product=product,
                    reason=reason,
                    review_summary=review_summary,
                )
                selected.append(chosen)
                for alternative in hits:
                    if alternative.product.product_id == product.product_id:
                        continue
                    rejected.append(
                        RejectedAlternative(
                            task_id=task.task_id,
                            product_id=alternative.product.product_id,
                            title=alternative.product.title,
                            reason=alternative_reason(product, alternative.product),
                        )
                    )
                    if sum(item.task_id == task.task_id for item in rejected) >= 2:
                        break
                break
            if chosen is None:
                gaps.append(
                    CritiqueIssue(
                        code="budget_or_tool_error",
                        message=f"Не удалось добавить товар для подзадачи «{task.query}»",
                        severity="error",
                        task_id=task.task_id,
                    )
                )
            if plan.intent is AgentIntent.MODIFY_CART:
                break

        current_ids = {item.product_id for item in cart.items}
        selected = [item for item in selected if item.product.product_id in current_ids]
        events.append(
            AgentEvent(
                event="node_finished",
                node="researcher",
                message=f"В корзине позиций {len(cart.items)}",
                data={
                    "subtotal": str(cart.subtotal),
                    "remaining": str(cart.remaining),
                    "tool_calls": len(records),
                    "gaps": len(gaps),
                },
            )
        )
        return {
            "cart": cart,
            "selected": selected,
            "rejected": rejected,
            "research_gaps": gaps,
            "tool_records": records,
            "events": events,
            "step_count": state["step_count"] + 1,
            "retry_research": False,
            "stop_requested": time.time() >= state["deadline_at"],
        }

    def _critic(self, state: AgentGraphState) -> dict[str, Any]:
        plan = state["plan"]
        assert plan is not None
        events = [
            *state["events"],
            AgentEvent(
                event="node_started",
                node="critic",
                message="Критик проверяет корзину и ограничения",
            ),
        ]
        issues = list(state["research_gaps"])
        try:
            normalized_cart = self.tools.normalize_cart(state["cart"])
            if not normalized_cart.within_budget:
                issues.append(
                    CritiqueIssue(
                        code="budget",
                        message="Корзина превышает бюджет",
                        severity="error",
                    )
                )
        except Exception as error:
            normalized_cart = state["cart"]
            issues.append(
                CritiqueIssue(
                    code="invalid_cart",
                    message=f"Корзина не прошла программную проверку: {error}",
                    severity="error",
                )
            )
        task_by_id = {task.task_id: task for task in plan.tasks}
        for selected in state["selected"]:
            try:
                actual = self.catalog.get_product(selected.product.product_id)
            except Exception:
                issues.append(
                    CritiqueIssue(
                        code="unknown_product",
                        message="В корзине найден неизвестный идентификатор товара",
                        severity="error",
                        product_id=selected.product.product_id,
                        task_id=selected.task_id,
                    )
                )
                continue
            task = task_by_id.get(selected.task_id)
            if task:
                filters = SearchFilters(
                    min_rating=plan.constraints.min_rating,
                    required_attributes=task.required_attributes,
                    excluded_terms=task.excluded_terms,
                    currency=plan.constraints.currency,
                )
                if not product_matches_filters(actual, filters):
                    issues.append(
                        CritiqueIssue(
                            code="product_constraint",
                            message=f"Товар {actual.product_id} нарушает фильтры подзадачи",
                            severity="error",
                            product_id=actual.product_id,
                            task_id=selected.task_id,
                        )
                    )
            summary = selected.review_summary
            if summary and any(not item.review_ids for item in [*summary.pros, *summary.cons]):
                issues.append(
                    CritiqueIssue(
                        code="review_evidence",
                        message="Утверждение об отзыве не имеет источника",
                        severity="error",
                        product_id=actual.product_id,
                        task_id=selected.task_id,
                    )
                )
        errors = [issue for issue in issues if issue.severity == "error"]
        correctable_codes = {"budget", "product_constraint", "unknown_product"}
        can_retry = (
            bool(errors)
            and all(issue.code in correctable_codes for issue in errors)
            and state["critic_revisions"] < self.settings.agent_max_critic_revisions
        )
        revisions = state["critic_revisions"] + int(can_retry)
        critique = CritiqueResult(
            approved=not errors,
            iteration=revisions,
            issues=issues,
        )
        blocked = [
            *state["blocked_product_ids"],
            *[issue.product_id for issue in errors if issue.product_id],
        ]
        events.append(
            AgentEvent(
                event="node_finished",
                node="critic",
                message="Проверка пройдена" if not errors else f"Найдено ошибок {len(errors)}",
                data={"approved": not errors, "retry": can_retry, "iteration": revisions},
            )
        )
        status = AgentStatus.COMPLETED if not errors else AgentStatus.DEGRADED
        return {
            "cart": normalized_cart,
            "critique": critique,
            "blocked_product_ids": list(dict.fromkeys(blocked)),
            "critic_revisions": revisions,
            "retry_research": can_retry,
            "status": status,
            "events": events,
            "step_count": state["step_count"] + 1,
        }

    @staticmethod
    def _after_critic(state: AgentGraphState) -> Literal["researcher", "finalizer"]:
        return "researcher" if state["retry_research"] else "finalizer"

    def _finalizer(self, state: AgentGraphState) -> dict[str, Any]:
        plan = state["plan"]
        assert plan is not None
        if plan.clarification_questions:
            status = AgentStatus.NEEDS_CLARIFICATION
            answer = " ".join(plan.clarification_questions)
        elif state["stop_requested"] and state["critique"] is None:
            status = AgentStatus.DEGRADED
            answer = "Лимит времени исчерпан до начала поиска. Попробуйте повторить запрос."
        elif state["critique"] and not state["critique"].approved:
            status = AgentStatus.DEGRADED
            errors = [
                issue.message for issue in state["critique"].issues if issue.severity == "error"
            ]
            answer = "Корзина собрана не полностью. " + " ".join(errors)
        else:
            status = AgentStatus.COMPLETED
            names = ", ".join(item.product.title for item in state["selected"])
            answer = (
                f"Корзина готова. Выбрано позиций {len(state['cart'].items)}. "
                f"Сумма {state['cart'].subtotal} {state['cart'].currency}. "
                f"Остаток {state['cart'].remaining} {state['cart'].currency}."
            )
            if names:
                answer += f" Выбраны {names}."
        events = [
            *state["events"],
            AgentEvent(
                event="node_finished",
                node="finalizer",
                message=answer,
                data={"status": status.value},
            ),
        ]
        return {
            "status": status,
            "answer": answer,
            "events": events,
            "step_count": state["step_count"] + 1,
        }

    def _tool_call(
        self,
        records: list[ToolCallRecord],
        name: str,
        task_id: str,
        call: Callable[[], ToolResult],
        product_ids: Callable[[ToolResult], list[str]],
    ) -> ToolResult | None:
        if len(records) >= self.settings.agent_max_tool_calls:
            return None
        started = time.perf_counter()
        try:
            with self.observability.span(name, "tool", {"task_id": task_id}) as span:
                result = call()
                ids = product_ids(result)
                self.observability.update(span, {"product_ids": ids})
            status: Literal["ok", "empty", "error"] = "ok" if ids else "empty"
            records.append(
                ToolCallRecord(
                    name=name,
                    task_id=task_id,
                    status=status,
                    duration_ms=int((time.perf_counter() - started) * 1000),
                    product_ids=ids,
                )
            )
            return result
        except Exception as error:
            records.append(
                ToolCallRecord(
                    name=name,
                    task_id=task_id,
                    status="error",
                    duration_ms=int((time.perf_counter() - started) * 1000),
                    error=str(error),
                )
            )
            return None


def alternative_reason(selected: Any, alternative: Any) -> str:
    reasons = ["расположен ниже в итоговой выдаче"]
    if alternative.price > selected.price:
        reasons.append("стоит дороже выбранного товара")
    if (
        selected.rating is not None
        and alternative.rating is not None
        and alternative.rating < selected.rating
    ):
        reasons.append("имеет более низкий рейтинг")
    return ", ".join(reasons)


def candidate_matches_task(query: str, product: Any) -> bool:
    normalized_query = query.casefold()
    searchable = product.searchable_text().casefold()
    product_roots = (
        "блендер",
        "пылесос",
        "кастрю",
        "сковор",
        "чайник",
        "бель",
        "одеял",
        "сетев",
        "удлинител",
    )
    required_roots = [root for root in product_roots if root in normalized_query]
    return not required_roots or any(root in searchable for root in required_roots)


def rank_modification_hits(message: str, old_product: Any, hits: list[Any]) -> list[Any]:
    normalized = message.casefold()
    if "мощн" in normalized:
        old_power = numeric_attribute(old_product, "мощность")
        if old_power is not None:
            hits = [
                hit
                for hit in hits
                if (numeric_attribute(hit.product, "мощность") or -1) > old_power
            ]
            return sorted(
                hits,
                key=lambda hit: -(numeric_attribute(hit.product, "мощность") or -1),
            )
    if any(word in normalized for word in ("тише", "потише")):
        old_noise = noise_level(old_product)
        hits = [hit for hit in hits if noise_level(hit.product) < old_noise]
        return sorted(hits, key=lambda hit: noise_level(hit.product))
    if "дешев" in normalized:
        hits = [hit for hit in hits if hit.product.price < old_product.price]
        return sorted(hits, key=lambda hit: hit.product.price)
    return hits


def numeric_attribute(product: Any, attribute: str) -> float | None:
    for key, value in product.attributes.items():
        if key.casefold() != attribute.casefold():
            continue
        match = re.search(r"\d+(?:[.,]\d+)?", value)
        return float(match.group(0).replace(",", ".")) if match else None
    return None


def noise_level(product: Any) -> int:
    value = " ".join(
        value.casefold() for key, value in product.attributes.items() if "шум" in key.casefold()
    )
    if "тих" in value:
        return 0
    if "обыч" in value:
        return 1
    if "шум" in value:
        return 2
    return 1
