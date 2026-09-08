from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from shopping_agent.config import get_settings
from shopping_agent.errors import (
    BudgetExceededError,
    CurrencyMismatchError,
    ProductNotFoundError,
)
from shopping_agent.models import (
    BudgetCalculatorInput,
    BudgetCalculatorOutput,
    CartOpsInput,
    CartOpsOutput,
    CompareProductsInput,
    CompareProductsOutput,
    DemoVerificationOutput,
    Product,
    Review,
    ReviewSummary,
    SearchCatalogInput,
    SearchCatalogOutput,
    SummarizeReviewsInput,
)
from shopping_agent.runtime import Runtime, build_runtime
from shopping_agent.verification import verify_demo


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.runtime = build_runtime(get_settings())
    yield


app = FastAPI(
    title="Shopping Cart Agent",
    version="0.1.0",
    description="Проверяемые инструменты каталога для агента подбора корзины",
    lifespan=lifespan,
)


def runtime(request: Request) -> Runtime:
    return request.app.state.runtime


@app.exception_handler(ProductNotFoundError)
def product_not_found(_: Request, error: ProductNotFoundError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(error)})


@app.exception_handler(CurrencyMismatchError)
def currency_mismatch(_: Request, error: CurrencyMismatchError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(error)})


@app.exception_handler(BudgetExceededError)
def budget_exceeded(_: Request, error: BudgetExceededError) -> JSONResponse:
    return JSONResponse(status_code=409, content={"detail": str(error)})


@app.get("/health")
def health(request: Request) -> dict[str, Any]:
    current = runtime(request)
    return {
        "status": "ok" if current.index_warning is None else "degraded",
        "catalog_products": current.catalog.count_products(),
        "indexed_products": current.indexed_products,
        "dense_index_ready": current.search_engine.vector_index.ready(),
        "warning": current.index_warning,
    }


@app.post("/admin/reindex")
def reindex(request: Request) -> dict[str, int]:
    current = runtime(request)
    indexed = current.search_engine.rebuild()
    current.indexed_products = indexed
    current.index_warning = None
    return {"indexed_products": indexed}


@app.post("/tools/search-catalog", response_model=SearchCatalogOutput)
def search_catalog(payload: SearchCatalogInput, request: Request) -> SearchCatalogOutput:
    return runtime(request).tools.search_catalog(payload)


@app.get("/tools/product/{product_id}", response_model=Product)
def get_product(product_id: str, request: Request) -> Product:
    return runtime(request).tools.get_product_details(product_id)


@app.get("/tools/product/{product_id}/reviews", response_model=list[Review])
def get_product_reviews(product_id: str, request: Request) -> list[Review]:
    return runtime(request).catalog.get_reviews(product_id)


@app.post("/tools/summarize-reviews", response_model=ReviewSummary)
def summarize_reviews(payload: SummarizeReviewsInput, request: Request) -> ReviewSummary:
    return runtime(request).tools.summarize_reviews(payload)


@app.post("/tools/compare-products", response_model=CompareProductsOutput)
def compare_products(payload: CompareProductsInput, request: Request) -> CompareProductsOutput:
    return runtime(request).tools.compare_products(payload)


@app.post("/tools/budget-calculator", response_model=BudgetCalculatorOutput)
def budget_calculator(payload: BudgetCalculatorInput, request: Request) -> BudgetCalculatorOutput:
    return runtime(request).tools.budget_calculator(payload)


@app.post("/tools/cart", response_model=CartOpsOutput)
def cart_ops(payload: CartOpsInput, request: Request) -> CartOpsOutput:
    return runtime(request).tools.cart_ops(payload)


@app.post("/demo/verify", response_model=DemoVerificationOutput)
def demo_verify(request: Request) -> DemoVerificationOutput:
    return verify_demo(runtime(request))
