from shopping_agent.models import (
    CartOpsInput,
    CartState,
    DemoChecks,
    DemoVerificationOutput,
    SearchCatalogInput,
    SummarizeReviewsInput,
)
from shopping_agent.runtime import Runtime


def verify_demo(current: Runtime) -> DemoVerificationOutput:
    blender_search = current.tools.search_catalog(
        SearchCatalogInput.model_validate(
            {
                "query": "тихий блендер для частой готовки",
                "filters": {"max_price": "8000", "min_rating": 4.0, "currency": "RUB"},
                "top_k": 3,
            }
        )
    )
    vacuum_search = current.tools.search_catalog(
        SearchCatalogInput.model_validate(
            {
                "query": "пылесос HEPA против пыли и шерсти",
                "filters": {
                    "categories": ["Уборка"],
                    "max_price": "15000",
                    "min_rating": 4.0,
                    "currency": "RUB",
                },
                "top_k": 3,
            }
        )
    )
    if not blender_search.hits or not vacuum_search.hits:
        raise RuntimeError("Демонстрационный поиск не нашёл товары")
    blender = blender_search.hits[0].product
    vacuum = vacuum_search.hits[0].product
    cart = CartState(budget="20000", currency="RUB")
    for product in (blender, vacuum):
        cart = current.tools.cart_ops(
            CartOpsInput(cart=cart, action="add", product_id=product.product_id)
        ).cart
    reviews = current.tools.summarize_reviews(
        SummarizeReviewsInput(product_id=blender.product_id, aspect="шум")
    )
    return DemoVerificationOutput(
        checks=DemoChecks(
            catalog_ids_only=all(item.product_id.startswith("DEMO-") for item in cart.items),
            within_budget=cart.within_budget,
            exact_arithmetic=cart.remaining == cart.budget - cart.subtotal,
            reviews_have_evidence=reviews.relevant_reviews > 0,
            hybrid_search_used=blender_search.retrieval_mode == "hybrid",
        ),
        selected=[blender, vacuum],
        cart=cart,
        review_summary=reviews,
        search_warnings=blender_search.warnings + vacuum_search.warnings,
    )
