from decimal import Decimal

import pytest

from shopping_agent.errors import BudgetExceededError, ProductNotFoundError
from shopping_agent.models import (
    BudgetCalculatorInput,
    CartOpsInput,
    CartState,
    CompareProductsInput,
    SummarizeReviewsInput,
)
from shopping_agent.runtime import Runtime
from shopping_agent.verification import verify_demo


def test_budget_uses_exact_decimal_arithmetic(runtime: Runtime) -> None:
    result = runtime.tools.budget_calculator(
        BudgetCalculatorInput(
            budget="1000.00",
            currency="RUB",
            items=[
                {"product_id": "A", "unit_price": "333.33", "quantity": 3},
                {"product_id": "B", "unit_price": "0.01", "quantity": 1},
            ],
        )
    )

    assert result.subtotal == Decimal("1000.00")
    assert result.remaining == Decimal("0.00")
    assert result.within_budget is True


def test_cart_add_and_replace_recalculate_total(runtime: Runtime) -> None:
    cart = CartState(budget="20000", currency="RUB")
    added = runtime.tools.cart_ops(
        CartOpsInput(cart=cart, action="add", product_id="DEMO-HK-001")
    ).cart
    replaced = runtime.tools.cart_ops(
        CartOpsInput(
            cart=added,
            action="replace",
            old_product_id="DEMO-HK-001",
            new_product_id="DEMO-HK-003",
        )
    ).cart

    assert added.subtotal == Decimal("6490")
    assert replaced.subtotal == Decimal("3490")
    assert replaced.items[0].product_id == "DEMO-HK-003"


def test_cart_rejects_budget_violation(runtime: Runtime) -> None:
    cart = CartState(budget="1000", currency="RUB")

    with pytest.raises(BudgetExceededError):
        runtime.tools.cart_ops(CartOpsInput(cart=cart, action="add", product_id="DEMO-HK-008"))


def test_cart_rejects_hallucinated_id(runtime: Runtime) -> None:
    cart = CartState(budget="20000", currency="RUB")

    with pytest.raises(ProductNotFoundError):
        runtime.tools.cart_ops(CartOpsInput(cart=cart, action="add", product_id="MADE-UP-PRODUCT"))


def test_review_summary_has_frequency_and_sources(runtime: Runtime) -> None:
    result = runtime.tools.summarize_reviews(
        SummarizeReviewsInput(product_id="DEMO-HK-001", aspect="шум")
    )

    assert result.relevant_reviews == 3
    assert result.pros[0].frequency == 2
    assert result.cons[0].frequency == 1
    assert set(result.pros[0].review_ids) == {"R-001-1", "R-001-2"}


def test_comparison_marks_missing_attributes(runtime: Runtime) -> None:
    result = runtime.tools.compare_products(
        CompareProductsInput(
            product_ids=["DEMO-HK-001", "DEMO-HK-003"],
            criteria=["цена", "мощность", "объём"],
        )
    )

    assert result.rows[0].values["мощность"] == "700 Вт"
    assert result.missing_values == {"DEMO-HK-003": ["объём"]}


def test_full_demo_verification(runtime: Runtime) -> None:
    result = verify_demo(runtime)

    assert all(result.checks.model_dump().values())
    assert result.cart.subtotal == Decimal("17980")
