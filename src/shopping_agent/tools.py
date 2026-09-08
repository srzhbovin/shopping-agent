from __future__ import annotations

import re
from collections import Counter
from decimal import Decimal

from shopping_agent.catalog import SQLiteCatalog
from shopping_agent.errors import BudgetExceededError, CurrencyMismatchError, ProductNotFoundError
from shopping_agent.models import (
    BudgetCalculatorInput,
    BudgetCalculatorOutput,
    BudgetLine,
    CartAction,
    CartItem,
    CartOpsInput,
    CartOpsOutput,
    CartState,
    CompareProductsInput,
    CompareProductsOutput,
    ComparisonRow,
    Product,
    Review,
    ReviewEvidence,
    ReviewSummary,
    SearchCatalogInput,
    SearchCatalogOutput,
    SummarizeReviewsInput,
)
from shopping_agent.retrieval import HybridSearchEngine

ASPECT_TERMS: dict[str, set[str]] = {
    "шум": {"шум", "шумный", "шумная", "тихий", "тихо", "noise", "noisy", "quiet"},
    "очистка": {"мыть", "моется", "чистить", "очистка", "wash", "clean", "cleaning"},
    "мощность": {"мощность", "мощный", "измельчает", "power", "powerful", "blend"},
    "аллергия": {"аллергия", "шерсть", "пыль", "hepa", "allergy", "hair", "dust"},
    "надёжность": {"сломался", "надёжный", "надёжность", "broken", "reliable", "durable"},
}


class ShoppingTools:
    def __init__(self, catalog: SQLiteCatalog, search_engine: HybridSearchEngine) -> None:
        self.catalog = catalog
        self.search_engine = search_engine

    def search_catalog(self, request: SearchCatalogInput) -> SearchCatalogOutput:
        return self.search_engine.search(request)

    def get_product_details(self, product_id: str) -> Product:
        return self.catalog.get_product(product_id)

    def summarize_reviews(self, request: SummarizeReviewsInput) -> ReviewSummary:
        reviews = self.catalog.get_reviews(request.product_id)
        relevant = [review for review in reviews if review_matches_aspect(review, request.aspect)]
        warnings: list[str] = []
        if request.aspect and not relevant:
            warnings.append("В отзывах нет подтверждений по запрошенному аспекту")
        pros = build_review_evidence([review for review in relevant if review.rating >= 4], True)
        cons = build_review_evidence([review for review in relevant if review.rating <= 3], False)
        return ReviewSummary(
            product_id=request.product_id,
            aspect=request.aspect,
            total_reviews=len(reviews),
            relevant_reviews=len(relevant),
            pros=pros,
            cons=cons,
            warnings=warnings,
        )

    def compare_products(self, request: CompareProductsInput) -> CompareProductsOutput:
        products = [self.catalog.get_product(product_id) for product_id in request.product_ids]
        rows: list[ComparisonRow] = []
        missing_values: dict[str, list[str]] = {}
        for product in products:
            values: dict[str, object] = {}
            for criterion in request.criteria:
                value = criterion_value(product, criterion)
                values[criterion] = value
                if value is None:
                    missing_values.setdefault(product.product_id, []).append(criterion)
            rows.append(ComparisonRow(product_id=product.product_id, values=values))
        return CompareProductsOutput(
            criteria=request.criteria,
            rows=rows,
            missing_values=missing_values,
        )

    def budget_calculator(self, request: BudgetCalculatorInput) -> BudgetCalculatorOutput:
        currency = request.currency.upper()
        subtotal = sum(
            (line.unit_price * line.quantity for line in request.items), start=Decimal("0")
        )
        remaining = request.budget - subtotal
        return BudgetCalculatorOutput(
            subtotal=subtotal,
            budget=request.budget,
            remaining=remaining,
            within_budget=remaining >= 0,
            currency=currency,
        )

    def cart_ops(self, request: CartOpsInput) -> CartOpsOutput:
        quantities = Counter({item.product_id: item.quantity for item in request.cart.items})
        if request.action is CartAction.ADD:
            assert request.product_id is not None
            self.catalog.get_product(request.product_id)
            quantities[request.product_id] += request.quantity
        elif request.action is CartAction.REMOVE:
            assert request.product_id is not None
            if request.product_id not in quantities:
                raise ProductNotFoundError(request.product_id)
            quantities[request.product_id] -= request.quantity
            if quantities[request.product_id] <= 0:
                del quantities[request.product_id]
        else:
            assert request.old_product_id is not None and request.new_product_id is not None
            if request.old_product_id not in quantities:
                raise ProductNotFoundError(request.old_product_id)
            self.catalog.get_product(request.new_product_id)
            previous_quantity = quantities.pop(request.old_product_id)
            quantities[request.new_product_id] += previous_quantity

        calculated = self._validated_cart(
            budget=request.cart.budget,
            currency=request.cart.currency,
            quantities=quantities,
        )
        if not calculated.within_budget:
            raise BudgetExceededError(
                f"Cart subtotal {calculated.subtotal} exceeds budget {calculated.budget}"
            )
        old_normalized = self._validated_cart(
            budget=request.cart.budget,
            currency=request.cart.currency,
            quantities=Counter({item.product_id: item.quantity for item in request.cart.items}),
        )
        return CartOpsOutput(
            cart=calculated,
            changed=calculated.items != old_normalized.items,
            operation=request.action.value,
        )

    def normalize_cart(self, cart: CartState) -> CartState:
        return self._validated_cart(
            budget=cart.budget,
            currency=cart.currency,
            quantities=Counter({item.product_id: item.quantity for item in cart.items}),
        )

    def _validated_cart(
        self, *, budget: Decimal, currency: str, quantities: Counter[str]
    ) -> CartState:
        normalized_currency = currency.upper()
        lines: list[BudgetLine] = []
        items: list[CartItem] = []
        for product_id in sorted(quantities):
            quantity = quantities[product_id]
            if quantity <= 0:
                continue
            product = self.catalog.get_product(product_id)
            if product.currency != normalized_currency:
                raise CurrencyMismatchError(
                    f"Product {product_id} uses {product.currency}, cart uses {normalized_currency}"
                )
            items.append(CartItem(product_id=product_id, quantity=quantity))
            lines.append(
                BudgetLine(product_id=product_id, unit_price=product.price, quantity=quantity)
            )
        calculation = self.budget_calculator(
            BudgetCalculatorInput(budget=budget, currency=normalized_currency, items=lines)
        )
        return CartState(
            budget=budget,
            currency=normalized_currency,
            items=items,
            subtotal=calculation.subtotal,
            remaining=calculation.remaining,
            within_budget=calculation.within_budget,
        )


def review_matches_aspect(review: Review, aspect: str | None) -> bool:
    if not aspect:
        return True
    normalized = aspect.casefold()
    terms = ASPECT_TERMS.get(normalized, {normalized}) | {normalized}
    text = f"{review.title} {review.text}".casefold()
    return any(term in text for term in terms)


def build_review_evidence(reviews: list[Review], positive: bool) -> list[ReviewEvidence]:
    if not reviews:
        return []
    representative = reviews[0].text.strip()
    if len(representative) > 240:
        representative = representative[:237].rstrip() + "..."
    return [
        ReviewEvidence(
            statement=representative,
            frequency=len(reviews),
            review_ids=[review.review_id for review in reviews[:10]],
        )
    ]


def criterion_value(product: Product, criterion: str) -> object | None:
    normalized = criterion.casefold()
    standard = {
        "цена": product.price,
        "price": product.price,
        "рейтинг": product.rating,
        "rating": product.rating,
        "число отзывов": product.review_count,
        "review_count": product.review_count,
        "категория": product.category,
        "category": product.category,
    }
    if normalized in standard:
        return standard[normalized]
    attribute_name = re.sub(r"^(attributes\.|характеристика\.)", "", normalized)
    for key, value in product.attributes.items():
        if key.casefold() == attribute_name:
            return value
    return None
