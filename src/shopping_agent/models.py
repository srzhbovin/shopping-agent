from __future__ import annotations

from decimal import Decimal
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Product(StrictModel):
    product_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    category: str = Field(min_length=1)
    description: str = ""
    price: Decimal = Field(ge=0)
    currency: str = Field(default="RUB", min_length=3, max_length=3)
    rating: float | None = Field(default=None, ge=0, le=5)
    review_count: int = Field(default=0, ge=0)
    attributes: dict[str, str] = Field(default_factory=dict)
    source_url: str | None = None

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, value: str) -> str:
        return value.upper()

    def searchable_text(self) -> str:
        attributes = " ".join(f"{key} {value}" for key, value in self.attributes.items())
        return " ".join((self.title, self.category, self.description, attributes))


class Review(StrictModel):
    review_id: str
    product_id: str
    rating: float = Field(ge=0, le=5)
    title: str = ""
    text: str = Field(min_length=1)
    verified_purchase: bool | None = None


class SearchFilters(StrictModel):
    min_price: Decimal | None = Field(default=None, ge=0)
    max_price: Decimal | None = Field(default=None, ge=0)
    categories: list[str] = Field(default_factory=list)
    min_rating: float | None = Field(default=None, ge=0, le=5)
    required_attributes: dict[str, str] = Field(default_factory=dict)
    excluded_terms: list[str] = Field(default_factory=list)
    currency: str | None = None

    @model_validator(mode="after")
    def validate_price_range(self) -> SearchFilters:
        if self.min_price is not None and self.max_price is not None:
            if self.min_price > self.max_price:
                raise ValueError("min_price must not exceed max_price")
        if self.currency:
            self.currency = self.currency.upper()
        return self


class SearchCatalogInput(StrictModel):
    query: str = Field(min_length=1)
    filters: SearchFilters = Field(default_factory=SearchFilters)
    top_k: int = Field(default=10, ge=1, le=50)


class SearchHit(StrictModel):
    product: Product
    score: float
    lexical_rank: int | None = None
    dense_rank: int | None = None
    matched_terms: list[str] = Field(default_factory=list)


class SearchCatalogOutput(StrictModel):
    query: str
    hits: list[SearchHit]
    retrieval_mode: str
    warnings: list[str] = Field(default_factory=list)


class ReviewEvidence(StrictModel):
    statement: str
    frequency: int = Field(ge=1)
    review_ids: list[str]


class SummarizeReviewsInput(StrictModel):
    product_id: str
    aspect: str | None = None


class ReviewSummary(StrictModel):
    product_id: str
    aspect: str | None
    total_reviews: int
    relevant_reviews: int
    pros: list[ReviewEvidence]
    cons: list[ReviewEvidence]
    warnings: list[str] = Field(default_factory=list)


class CompareProductsInput(StrictModel):
    product_ids: list[str] = Field(min_length=2, max_length=10)
    criteria: list[str] = Field(min_length=1, max_length=20)


class ComparisonRow(StrictModel):
    product_id: str
    values: dict[str, Any]


class CompareProductsOutput(StrictModel):
    criteria: list[str]
    rows: list[ComparisonRow]
    missing_values: dict[str, list[str]]


class BudgetLine(StrictModel):
    product_id: str
    unit_price: Decimal = Field(ge=0)
    quantity: int = Field(default=1, ge=1, le=100)


class BudgetCalculatorInput(StrictModel):
    budget: Decimal = Field(ge=0)
    currency: str
    items: list[BudgetLine]


class BudgetCalculatorOutput(StrictModel):
    subtotal: Decimal
    budget: Decimal
    remaining: Decimal
    within_budget: bool
    currency: str


class CartItem(StrictModel):
    product_id: str
    quantity: int = Field(default=1, ge=1, le=100)


class CartState(StrictModel):
    budget: Decimal = Field(ge=0)
    currency: str
    items: list[CartItem] = Field(default_factory=list)
    subtotal: Decimal = Field(default=Decimal("0"), ge=0)
    remaining: Decimal = Decimal("0")
    within_budget: bool = True


class CartAction(StrEnum):
    ADD = "add"
    REMOVE = "remove"
    REPLACE = "replace"


class CartOpsInput(StrictModel):
    cart: CartState
    action: CartAction
    product_id: str | None = None
    quantity: int = Field(default=1, ge=1, le=100)
    old_product_id: str | None = None
    new_product_id: str | None = None

    @model_validator(mode="after")
    def validate_action_fields(self) -> CartOpsInput:
        if self.action in {CartAction.ADD, CartAction.REMOVE} and not self.product_id:
            raise ValueError("product_id is required for add and remove")
        if self.action is CartAction.REPLACE and not (self.old_product_id and self.new_product_id):
            raise ValueError("old_product_id and new_product_id are required for replace")
        return self


class CartOpsOutput(StrictModel):
    cart: CartState
    changed: bool
    operation: Literal["add", "remove", "replace"]


class DemoChecks(StrictModel):
    catalog_ids_only: bool
    within_budget: bool
    exact_arithmetic: bool
    reviews_have_evidence: bool
    hybrid_search_used: bool


class DemoVerificationOutput(StrictModel):
    checks: DemoChecks
    selected: list[Product]
    cart: CartState
    review_summary: ReviewSummary
    search_warnings: list[str]
