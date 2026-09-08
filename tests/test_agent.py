from shopping_agent.agent_models import (
    AgentRequest,
    AgentResponse,
    AgentStatus,
    HardConstraints,
    SessionSnapshot,
)
from shopping_agent.models import CartState, SearchCatalogOutput, SearchHit
from shopping_agent.runtime import Runtime


def test_agent_builds_verified_cart_and_saves_trace(runtime: Runtime) -> None:
    result = runtime.agent_service.run(
        AgentRequest(
            message=(
                "Собери набор для первого переезда в квартиру, бюджет 40000 рублей, "
                "готовлю часто, аллергия на шерсть"
            )
        )
    )

    assert result.status is AgentStatus.COMPLETED
    assert result.plan.planner_mode == "rules"
    assert result.cart.within_budget
    assert result.cart.subtotal == 30840
    assert len(result.cart.items) == 6
    assert result.critique is not None and result.critique.approved
    assert result.metrics.step_count == 4
    assert result.metrics.tool_calls > 6
    catalog_ids = {product.product_id for product in runtime.catalog.list_products()}
    assert {item.product_id for item in result.cart.items} <= catalog_ids
    assert all(item.reason for item in result.selected)
    assert all(
        evidence.review_ids
        for item in result.selected
        if item.review_summary
        for evidence in [*item.review_summary.pros, *item.review_summary.cons]
    )

    snapshot = runtime.agent_service.get_session(result.session_id)
    trace = runtime.agent_service.get_trace(result.trace_id)
    assert snapshot is not None and snapshot.cart == result.cart
    assert trace is not None and trace.status == "completed"
    assert len(trace.tool_calls) == result.metrics.tool_calls
    assert [event.node for event in trace.events] == [
        "planner",
        "planner",
        "researcher",
        "researcher",
        "critic",
        "critic",
        "finalizer",
    ]


def test_agent_asks_once_for_budget_and_continues_session(runtime: Runtime) -> None:
    first = runtime.agent_service.run(
        AgentRequest(message="Собери набор для первого переезда в квартиру")
    )

    assert first.status is AgentStatus.NEEDS_CLARIFICATION
    assert first.clarification_questions == ["Какой максимальный бюджет корзины?"]
    assert first.metrics.tool_calls == 0

    second = runtime.agent_service.run(AgentRequest(message="40000", session_id=first.session_id))

    assert second.status is AgentStatus.COMPLETED
    assert second.session_id == first.session_id
    assert second.plan.constraints.budget == 40000
    assert second.cart.items
    snapshot = runtime.agent_service.get_session(first.session_id)
    assert snapshot is not None and snapshot.clarification_count == 1


def test_agent_replaces_one_item_without_rebuilding_cart(runtime: Runtime) -> None:
    first = runtime.agent_service.run(AgentRequest(message="Подбери тихий блендер до 8000 рублей"))
    first_ids = {item.product_id for item in first.cart.items}
    assert first_ids == {"DEMO-HK-001"}

    second = runtime.agent_service.run(
        AgentRequest(
            message="Замени блендер на более мощный",
            session_id=first.session_id,
        )
    )
    second_ids = {item.product_id for item in second.cart.items}

    assert second.status is AgentStatus.COMPLETED
    assert second.plan.intent.value == "modify_cart"
    assert second.plan.target_product_id == "DEMO-HK-001"
    assert "DEMO-HK-001" not in second_ids
    assert second_ids == {"DEMO-HK-002"}


def test_agent_keeps_item_when_quieter_alternative_does_not_exist(
    runtime: Runtime,
) -> None:
    first = runtime.agent_service.run(AgentRequest(message="Подбери тихий блендер до 8000 рублей"))
    second = runtime.agent_service.run(
        AgentRequest(
            message="Замени блендер на что-то потише",
            session_id=first.session_id,
        )
    )

    assert second.status is AgentStatus.DEGRADED
    assert {item.product_id for item in second.cart.items} == {"DEMO-HK-001"}
    assert any(issue.code == "no_result" for issue in second.critique.issues)


def test_agent_reports_when_no_catalog_product_matches(runtime: Runtime) -> None:
    result = runtime.agent_service.run(AgentRequest(message="Подбери блендер до 3000 рублей"))

    assert result.status is AgentStatus.DEGRADED
    assert result.cart.items == []
    assert result.critique is not None
    assert any(issue.code == "no_result" for issue in result.critique.issues)


def test_agent_does_not_repeat_known_budget_and_enforces_bowl_material(
    runtime: Runtime,
) -> None:
    result = runtime.agent_service.run(
        AgentRequest(message="Подбери тихий блендер с тритановой чашей в пределах 7500 рублей")
    )

    assert result.status is AgentStatus.COMPLETED
    assert result.clarification_questions == []
    assert result.plan.constraints.budget == 7500
    assert result.plan.tasks[0].required_attributes == {
        "режим шума": "тихий",
        "материал чаши": "тритан",
    }
    assert result.selected[0].product.product_id == "DEMO-HK-001"


def test_agent_stream_contains_progress_and_result(runtime: Runtime) -> None:
    items = list(
        runtime.agent_service.stream(AgentRequest(message="Подбери чайник до 4000 рублей"))
    )

    assert isinstance(items[-1], AgentResponse)
    events = [item for item in items[:-1] if not isinstance(item, AgentResponse)]
    assert {event.node for event in events} >= {"planner", "researcher", "critic"}


def test_critic_stops_after_two_revisions(runtime: Runtime, monkeypatch) -> None:
    unsuitable = [
        runtime.catalog.get_product("DEMO-HK-003"),
        runtime.catalog.get_product("DEMO-HK-002"),
    ]

    def invalid_search(request):
        return SearchCatalogOutput(
            query=request.query,
            hits=[
                SearchHit(product=product, score=1 - index * 0.1)
                for index, product in enumerate(unsuitable)
            ],
            retrieval_mode="test",
        )

    monkeypatch.setattr(runtime.tools, "search_catalog", invalid_search)
    result = runtime.agent_service.run(AgentRequest(message="Подбери тихий блендер до 8000 рублей"))

    assert result.status is AgentStatus.DEGRADED
    assert result.metrics.critic_revisions == 2
    assert result.critique is not None and not result.critique.approved
    critic_finishes = [
        event
        for event in runtime.agent_service.get_trace(result.trace_id).events
        if event.node == "critic" and event.event == "node_finished"
    ]
    assert len(critic_finishes) == 3


def test_session_token_budget_degrades_to_rules_without_llm(runtime: Runtime) -> None:
    session_id = "session-token-budget"
    runtime.agent_store.save_session(
        SessionSnapshot(
            session_id=session_id,
            original_request="Подбери чайник до 4000 рублей",
            constraints=HardConstraints(budget=4000),
            cart=CartState(budget=4000, currency="RUB", remaining=4000),
            cumulative_input_tokens=900,
            cumulative_output_tokens=300,
        )
    )

    result = runtime.agent_service.run(
        AgentRequest(message="Подбери чайник", session_id=session_id)
    )

    assert result.plan.planner_mode == "rules"
    assert result.metrics.llm_calls == 0
    assert result.metrics.session_tokens_used == 1200
    assert "Исчерпан токен-бюджет сессии" in result.metrics.degraded_reasons
