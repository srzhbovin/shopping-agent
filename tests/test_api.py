import asyncio

import httpx

from shopping_agent.api import app
from shopping_agent.runtime import Runtime


def test_http_api_exposes_tools_and_errors(runtime: Runtime) -> None:
    async def scenario() -> None:
        app.state.runtime = runtime
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=True)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            health = await client.get("/health")
            assert health.status_code == 200
            assert health.json()["catalog_products"] == 12

            search = await client.post(
                "/tools/search-catalog",
                json={
                    "query": "тихий блендер",
                    "filters": {"max_price": "8000", "currency": "RUB"},
                    "top_k": 3,
                },
            )
            assert search.status_code == 200
            assert search.json()["hits"][0]["product"]["product_id"] == "DEMO-HK-001"

            reviews = await client.get("/tools/product/DEMO-HK-001/reviews")
            assert reviews.status_code == 200
            assert len(reviews.json()) == 3

            missing = await client.get("/tools/product/UNKNOWN")
            assert missing.status_code == 404

            rejected = await client.post(
                "/tools/cart",
                json={
                    "cart": {
                        "budget": "1000",
                        "currency": "RUB",
                        "items": [],
                        "subtotal": "0",
                        "remaining": "1000",
                        "within_budget": True,
                    },
                    "action": "add",
                    "product_id": "DEMO-HK-008",
                    "quantity": 1,
                },
            )
            assert rejected.status_code == 409

            verification = await client.post("/demo/verify")
            assert verification.status_code == 200
            assert all(verification.json()["checks"].values())

    asyncio.run(scenario())


def test_http_api_runs_agent_stream_and_exposes_state(runtime: Runtime) -> None:
    async def scenario() -> None:
        app.state.runtime = runtime
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=True)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/agent/run",
                json={"message": "Подбери тихий блендер до 8000 рублей"},
            )
            assert response.status_code == 200
            result = response.json()
            assert result["status"] == "completed"
            assert result["cart"]["items"][0]["product_id"] == "DEMO-HK-001"

            session = await client.get(f"/agent/sessions/{result['session_id']}")
            trace = await client.get(f"/agent/traces/{result['trace_id']}")
            assert session.status_code == 200
            assert trace.status_code == 200
            assert trace.json()["events"][0]["node"] == "planner"

            streamed = await client.post(
                "/agent/stream",
                json={"message": "Подбери чайник до 4000 рублей"},
            )
            assert streamed.status_code == 200
            assert "event: accepted" in streamed.text
            assert "event: progress" in streamed.text
            assert "event: result" in streamed.text

            assert (await client.get("/agent/sessions/missing-session")).status_code == 404
            assert (await client.get("/agent/traces/missing-trace")).status_code == 404

    asyncio.run(scenario())
