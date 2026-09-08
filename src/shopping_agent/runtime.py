from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from shopping_agent.catalog import SQLiteCatalog
from shopping_agent.config import Settings
from shopping_agent.demo_data import load_demo_data
from shopping_agent.embeddings import CachedEmbedder, Embedder, HashingEmbedder, LMStudioEmbedder
from shopping_agent.retrieval import HybridSearchEngine
from shopping_agent.tools import ShoppingTools
from shopping_agent.vector_index import InMemoryVectorIndex, QdrantVectorIndex, VectorIndex


@dataclass
class Runtime:
    catalog: SQLiteCatalog
    search_engine: HybridSearchEngine
    tools: ShoppingTools
    indexed_products: int
    index_warning: str | None = None


def build_runtime(
    settings: Settings,
    *,
    embedder: Embedder | None = None,
    vector_index: VectorIndex | None = None,
    demo_data_dir: Path | None = None,
) -> Runtime:
    catalog = SQLiteCatalog(settings.catalog_db_path)
    if settings.auto_load_demo and catalog.count_products() == 0:
        data_dir = demo_data_dir or settings.demo_data_path
        load_demo_data(catalog, data_dir)
    if embedder is None:
        embedder = CachedEmbedder(
            LMStudioEmbedder(
                settings.embedding_base_url,
                settings.embedding_model,
                settings.embedding_timeout_seconds,
            ),
            catalog,
        )
    if vector_index is None:
        vector_index = QdrantVectorIndex(settings.qdrant_url, settings.qdrant_collection)
    search_engine = HybridSearchEngine(catalog, embedder, vector_index)
    warning = None
    indexed = 0
    try:
        indexed = search_engine.rebuild()
    except Exception as error:
        warning = str(error)
        # BM25 must remain available when LM Studio or Qdrant is temporarily unavailable.
        search_engine._products = catalog.list_products()
        from shopping_agent.retrieval import BM25Index

        search_engine._bm25 = BM25Index(search_engine._products)
    return Runtime(
        catalog=catalog,
        search_engine=search_engine,
        tools=ShoppingTools(catalog, search_engine),
        indexed_products=indexed,
        index_warning=warning,
    )


def build_test_runtime(settings: Settings, demo_data_dir: Path) -> Runtime:
    return build_runtime(
        settings,
        embedder=HashingEmbedder(),
        vector_index=InMemoryVectorIndex(),
        demo_data_dir=demo_data_dir,
    )
