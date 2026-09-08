from shopping_agent.embeddings import HashingEmbedder
from shopping_agent.models import SearchCatalogInput
from shopping_agent.retrieval import HybridSearchEngine
from shopping_agent.runtime import Runtime
from shopping_agent.vector_index import InMemoryVectorIndex


def test_hybrid_search_finds_quiet_blender(runtime: Runtime) -> None:
    result = runtime.tools.search_catalog(
        SearchCatalogInput(
            query="тихий блендер для готовки",
            filters={"max_price": "8000", "min_rating": 4.0, "currency": "RUB"},
            top_k=3,
        )
    )

    assert result.retrieval_mode == "hybrid"
    assert result.hits[0].product.product_id == "DEMO-HK-001"
    assert all(hit.product.price <= 8000 for hit in result.hits)


def test_filters_remove_glass_products(runtime: Runtime) -> None:
    result = runtime.tools.search_catalog(
        SearchCatalogInput(
            query="набор кастрюль для ежедневной готовки",
            filters={"categories": ["Посуда"], "excluded_terms": ["стекло"]},
            top_k=10,
        )
    )

    ids = {hit.product.product_id for hit in result.hits}
    assert "DEMO-HK-005" not in ids
    assert "DEMO-HK-006" in ids


def test_unknown_product_is_never_returned(runtime: Runtime) -> None:
    result = runtime.tools.search_catalog(SearchCatalogInput(query="пылесос", top_k=20))
    catalog_ids = {product.product_id for product in runtime.catalog.list_products()}

    assert result.hits
    assert {hit.product.product_id for hit in result.hits} <= catalog_ids


def test_empty_filtered_result_is_handled(runtime: Runtime) -> None:
    result = runtime.tools.search_catalog(
        SearchCatalogInput(query="блендер", filters={"max_price": "1"}, top_k=10)
    )

    assert result.hits == []


def test_dense_failure_falls_back_to_bm25(runtime: Runtime) -> None:
    class FailingEmbedder(HashingEmbedder):
        def embed(self, texts, *, kind):  # type: ignore[no-untyped-def]
            if kind == "query":
                raise TimeoutError("embedding timeout")
            return super().embed(texts, kind=kind)

    engine = HybridSearchEngine(runtime.catalog, FailingEmbedder(), InMemoryVectorIndex())
    engine.rebuild()
    result = engine.search(SearchCatalogInput(query="тихий блендер", top_k=3))

    assert result.retrieval_mode == "bm25"
    assert result.hits[0].product.product_id == "DEMO-HK-001"
    assert "embedding timeout" in result.warnings[0]
