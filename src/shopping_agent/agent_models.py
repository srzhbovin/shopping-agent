from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any, Literal

from pydantic import Field, field_validator

from shopping_agent.models import CartState, Product, ReviewSummary, StrictModel


class AgentIntent(StrEnum):
    BUILD_CART = "build_cart"
    MODIFY_CART = "modify_cart"


class AgentStatus(StrEnum):
    COMPLETED = "completed"
    NEEDS_CLARIFICATION = "needs_clarification"
    DEGRADED = "degraded"


class HardConstraints(StrictModel):
    budget: Decimal | None = Field(default=None, ge=0)
    currency: str = Field(default="RUB", min_length=3, max_length=3)
    min_rating: float | None = Field(default=None, ge=0, le=5)
    excluded_terms: list[str] = Field(default_factory=list)
    must_have: list[str] = Field(default_factory=list)
    deal_breakers: list[str] = Field(default_factory=list)

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, value: str) -> str:
        return value.upper()


class ResearchTask(StrictModel):
    task_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    category: str | None = None
    required_attributes: dict[str, str] = Field(default_factory=dict)
    excluded_terms: list[str] = Field(default_factory=list)
    criteria: list[str] = Field(default_factory=lambda: ["цена", "рейтинг"])
    review_aspect: str | None = None
    priority: int = Field(default=5, ge=1, le=10)


class ShoppingPlan(StrictModel):
    intent: AgentIntent
    constraints: HardConstraints
    preferences: list[str] = Field(default_factory=list)
    tasks: list[ResearchTask] = Field(default_factory=list, max_length=8)
    target_product_id: str | None = None
    clarification_questions: list[str] = Field(default_factory=list, max_length=2)
    assumptions: list[str] = Field(default_factory=list)
    planner_mode: Literal["llm", "cache", "rules"] = "llm"


class PlannerDraft(StrictModel):
    intent: AgentIntent = AgentIntent.BUILD_CART
    budget: Decimal | None = Field(default=None, ge=0)
    currency: str = "RUB"
    min_rating: float | None = Field(default=None, ge=0, le=5)
    excluded_terms: list[str] = Field(default_factory=list)
    preferences: list[str] = Field(default_factory=list)
    tasks: list[str] = Field(default_factory=list, max_length=6)
    target_terms: str | None = None
    questions: list[str] = Field(default_factory=list, max_length=2)


class SelectedItem(StrictModel):
    task_id: str
    product: Product
    reason: str
    review_summary: ReviewSummary | None = None


class RejectedAlternative(StrictModel):
    task_id: str
    product_id: str
    title: str
    reason: str


class CritiqueIssue(StrictModel):
    code: str
    message: str
    severity: Literal["error", "warning"]
    product_id: str | None = None
    task_id: str | None = None


class CritiqueResult(StrictModel):
    approved: bool
    iteration: int = Field(ge=0, le=2)
    issues: list[CritiqueIssue] = Field(default_factory=list)


class ToolCallRecord(StrictModel):
    name: str
    task_id: str | None = None
    status: Literal["ok", "error", "empty"]
    duration_ms: int = Field(ge=0)
    product_ids: list[str] = Field(default_factory=list)
    error: str | None = None


class RunMetrics(StrictModel):
    duration_seconds: float = Field(ge=0)
    step_count: int = Field(ge=0)
    tool_calls: int = Field(ge=0)
    llm_calls: int = Field(ge=0)
    llm_cache_hits: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    session_tokens_used: int = Field(default=0, ge=0)
    critic_revisions: int = Field(ge=0, le=2)
    degraded_reasons: list[str] = Field(default_factory=list)
    langfuse_enabled: bool = False


class AgentRequest(StrictModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(default=None, min_length=8, max_length=100)


class AgentResponse(StrictModel):
    session_id: str
    trace_id: str
    status: AgentStatus
    message: str
    plan: ShoppingPlan
    cart: CartState
    selected: list[SelectedItem]
    rejected_alternatives: list[RejectedAlternative]
    critique: CritiqueResult | None
    clarification_questions: list[str]
    metrics: RunMetrics


class AgentEvent(StrictModel):
    event: str
    node: str
    message: str
    data: dict[str, Any] = Field(default_factory=dict)


class SessionSnapshot(StrictModel):
    session_id: str
    original_request: str
    constraints: HardConstraints
    plan: ShoppingPlan | None = None
    cart: CartState
    selected: list[SelectedItem] = Field(default_factory=list)
    rejected_alternatives: list[RejectedAlternative] = Field(default_factory=list)
    clarification_count: int = Field(default=0, ge=0, le=2)
    cumulative_input_tokens: int = Field(default=0, ge=0)
    cumulative_output_tokens: int = Field(default=0, ge=0)
    last_trace_id: str | None = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class TraceRecord(StrictModel):
    trace_id: str
    session_id: str
    status: str
    input_message: str
    events: list[AgentEvent]
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    metrics: RunMetrics | None = None
    started_at: datetime
    finished_at: datetime | None = None
