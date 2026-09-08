from __future__ import annotations

import argparse
import json

import httpx


def verify() -> int:
    from shopping_agent.config import get_settings
    from shopping_agent.runtime import build_runtime
    from shopping_agent.verification import verify_demo

    current = build_runtime(get_settings())
    result = verify_demo(current)
    payload = result.model_dump(mode="json")
    health = {
        "catalog_products": current.catalog.count_products(),
        "indexed_products": current.indexed_products,
        "index_warning": current.index_warning,
    }
    print(json.dumps({"health": health, "verification": payload}, ensure_ascii=False, indent=2))
    return 0 if all(result.checks.model_dump().values()) else 1


def reset_demo() -> int:
    from shopping_agent.config import get_settings

    path = get_settings().catalog_db_path.resolve()
    if path.exists():
        path.unlink()
        print(f"Удалён файл демонстрационного каталога {path}")
    return 0


def check_models() -> int:
    from shopping_agent.config import get_settings

    settings = get_settings()
    embedding_response = httpx.post(
        f"{settings.embedding_base_url.rstrip('/')}/embeddings",
        json={"model": settings.embedding_model, "input": ["search_query: проверка"]},
        timeout=settings.embedding_timeout_seconds,
    )
    embedding_response.raise_for_status()
    dimensions = len(embedding_response.json()["data"][0]["embedding"])
    chat_response = httpx.post(
        f"{settings.llm_base_url.rstrip('/')}/chat",
        json={
            "model": settings.llm_model,
            "input": "Return exactly the word OK",
            "reasoning": "off",
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "max_output_tokens": 40,
            "store": False,
        },
        timeout=180,
    )
    chat_response.raise_for_status()
    payload = chat_response.json()
    messages = [item["content"] for item in payload["output"] if item["type"] == "message"]
    result = {
        "embedding_model": settings.embedding_model,
        "embedding_dimensions": dimensions,
        "llm_model": settings.llm_model,
        "llm_answer": "".join(messages),
        "reasoning_tokens": payload["stats"]["reasoning_output_tokens"],
        "tokens_per_second": payload["stats"]["tokens_per_second"],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["llm_answer"].strip() == "OK" and dimensions > 0 else 1


def verify_agent() -> int:
    from shopping_agent.agent_models import AgentRequest
    from shopping_agent.config import get_settings
    from shopping_agent.runtime import build_runtime

    current = build_runtime(get_settings())
    result = current.agent_service.run(
        AgentRequest(
            message=(
                "Собери набор для первого переезда в квартиру, бюджет 40000 рублей, "
                "готовлю часто, аллергия на шерсть"
            )
        )
    )
    catalog_ids = {product.product_id for product in current.catalog.list_products()}
    checks = {
        "planner_used_local_model": result.plan.planner_mode in {"llm", "cache"},
        "catalog_ids_only": all(item.product_id in catalog_ids for item in result.cart.items),
        "within_budget": result.cart.within_budget,
        "critic_ran": result.critique is not None,
        "trace_saved": current.agent_service.get_trace(result.trace_id) is not None,
    }
    print(
        json.dumps(
            {"checks": checks, "result": result.model_dump(mode="json")},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if all(checks.values()) else 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["verify", "verify-agent", "reset-demo", "check-models"])
    args = parser.parse_args()
    commands = {
        "verify": verify,
        "verify-agent": verify_agent,
        "reset-demo": reset_demo,
        "check-models": check_models,
    }
    raise SystemExit(commands[args.command]())


if __name__ == "__main__":
    main()
