from __future__ import annotations

import time
from collections.abc import Iterator
from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

from shopping_agent.agent_graph import AgentGraphState, ShoppingAgentGraph
from shopping_agent.agent_models import (
    AgentEvent,
    AgentIntent,
    AgentRequest,
    AgentResponse,
    AgentStatus,
    HardConstraints,
    RunMetrics,
    SessionSnapshot,
    ShoppingPlan,
    TraceRecord,
)
from shopping_agent.agent_store import AgentStore
from shopping_agent.config import Settings
from shopping_agent.models import CartState
from shopping_agent.tracing import LangfuseBridge


class AgentService:
    def __init__(
        self,
        *,
        graph: ShoppingAgentGraph,
        store: AgentStore,
        settings: Settings,
        observability: LangfuseBridge,
    ) -> None:
        self.graph = graph
        self.store = store
        self.settings = settings
        self.observability = observability

    def run(self, request: AgentRequest) -> AgentResponse:
        response: AgentResponse | None = None
        for item in self.stream(request):
            if isinstance(item, AgentResponse):
                response = item
        if response is None:
            raise RuntimeError("Граф не вернул итоговый ответ")
        return response

    def stream(self, request: AgentRequest) -> Iterator[AgentEvent | AgentResponse]:
        started_clock = time.perf_counter()
        started_at = datetime.now(UTC)
        session_id = request.session_id or f"session-{uuid4().hex}"
        trace_id = f"trace-{uuid4().hex}"
        previous = self.store.load_session(session_id)
        state = self._initial_state(request, session_id, trace_id, previous)
        seen_events = 0
        with self.observability.trace(
            session_id=session_id,
            trace_id=trace_id,
            input_message=request.message,
        ) as root_span:
            try:
                chunks = self.graph.graph.stream(
                    state,
                    config={"recursion_limit": self.settings.agent_max_steps},
                    stream_mode="updates",
                )
                for chunk in chunks:
                    for update in chunk.values():
                        state.update(update)
                        events = state.get("events", [])
                        for event in events[seen_events:]:
                            yield event
                        seen_events = len(events)
            except Exception as error:
                state = self._degraded_state(state, error)
                for event in state["events"][seen_events:]:
                    yield event
            response = self._finish_response(state, started_clock)
            self._persist(request, previous, response, state, started_at)
            self.observability.update(root_span, response.model_dump(mode="json"))
            self.observability.flush()
            yield response

    def get_session(self, session_id: str) -> SessionSnapshot | None:
        return self.store.load_session(session_id)

    def get_trace(self, trace_id: str) -> TraceRecord | None:
        return self.store.load_trace(trace_id)

    def _initial_state(
        self,
        request: AgentRequest,
        session_id: str,
        trace_id: str,
        previous: SessionSnapshot | None,
    ) -> AgentGraphState:
        base_cart = (
            previous.cart
            if previous
            else CartState(budget=Decimal("0"), currency="RUB", remaining=Decimal("0"))
        )
        base_selected = list(previous.selected) if previous else []
        return AgentGraphState(
            request=request,
            session_id=session_id,
            trace_id=trace_id,
            previous=previous,
            plan=None,
            base_cart=base_cart,
            base_selected=base_selected,
            cart=base_cart,
            selected=base_selected,
            rejected=[],
            critique=None,
            tool_records=[],
            events=[],
            research_gaps=[],
            blocked_product_ids=[],
            step_count=0,
            llm_calls=0,
            llm_cache_hits=0,
            input_tokens=0,
            output_tokens=0,
            session_tokens_used=(
                previous.cumulative_input_tokens + previous.cumulative_output_tokens
                if previous
                else 0
            ),
            critic_revisions=0,
            retry_research=False,
            status=AgentStatus.DEGRADED,
            answer="",
            degraded_reasons=[],
            deadline_at=time.time() + self.settings.agent_timeout_seconds,
            stop_requested=False,
        )

    def _degraded_state(self, state: AgentGraphState, error: Exception) -> AgentGraphState:
        reason = f"Выполнение остановлено безопасно: {error}"
        plan = state.get("plan")
        if plan is None:
            budget = state["previous"].constraints.budget if state["previous"] else None
            plan = ShoppingPlan(
                intent=AgentIntent.BUILD_CART,
                constraints=HardConstraints(budget=budget),
                clarification_questions=(
                    ["Какой максимальный бюджет корзины?"] if budget is None else []
                ),
                planner_mode="rules",
            )
        state.update(
            plan=plan,
            status=AgentStatus.DEGRADED,
            answer=reason,
            degraded_reasons=[*state["degraded_reasons"], reason],
            events=[
                *state["events"],
                AgentEvent(
                    event="run_degraded",
                    node="runtime",
                    message=reason,
                ),
            ],
        )
        return state

    def _finish_response(self, state: AgentGraphState, started_clock: float) -> AgentResponse:
        plan = state["plan"]
        assert plan is not None
        duration = time.perf_counter() - started_clock
        degraded = list(dict.fromkeys(state["degraded_reasons"]))
        if state["status"] is AgentStatus.DEGRADED and not degraded:
            degraded = ["Не все подзадачи удалось выполнить"]
        metrics = RunMetrics(
            duration_seconds=round(duration, 3),
            step_count=state["step_count"],
            tool_calls=len(state["tool_records"]),
            llm_calls=state["llm_calls"],
            llm_cache_hits=state["llm_cache_hits"],
            input_tokens=state["input_tokens"],
            output_tokens=state["output_tokens"],
            session_tokens_used=state["session_tokens_used"],
            critic_revisions=state["critic_revisions"],
            degraded_reasons=degraded,
            langfuse_enabled=self.observability.enabled,
        )
        return AgentResponse(
            session_id=state["session_id"],
            trace_id=state["trace_id"],
            status=state["status"],
            message=state["answer"],
            plan=plan,
            cart=state["cart"],
            selected=state["selected"],
            rejected_alternatives=state["rejected"],
            critique=state["critique"],
            clarification_questions=plan.clarification_questions,
            metrics=metrics,
        )

    def _persist(
        self,
        request: AgentRequest,
        previous: SessionSnapshot | None,
        response: AgentResponse,
        state: AgentGraphState,
        started_at: datetime,
    ) -> None:
        previous_count = previous.clarification_count if previous else 0
        asked = len(response.clarification_questions)
        snapshot = SessionSnapshot(
            session_id=response.session_id,
            original_request=previous.original_request if previous else request.message,
            constraints=response.plan.constraints,
            plan=response.plan,
            cart=response.cart,
            selected=response.selected,
            rejected_alternatives=response.rejected_alternatives,
            clarification_count=min(2, previous_count + asked),
            cumulative_input_tokens=(
                (previous.cumulative_input_tokens if previous else 0)
                + response.metrics.input_tokens
            ),
            cumulative_output_tokens=(
                (previous.cumulative_output_tokens if previous else 0)
                + response.metrics.output_tokens
            ),
            last_trace_id=response.trace_id,
        )
        self.store.save_session(snapshot)
        self.store.save_trace(
            TraceRecord(
                trace_id=response.trace_id,
                session_id=response.session_id,
                status=response.status.value,
                input_message=request.message,
                events=state["events"],
                tool_calls=state["tool_records"],
                metrics=response.metrics,
                started_at=started_at,
                finished_at=datetime.now(UTC),
            )
        )
