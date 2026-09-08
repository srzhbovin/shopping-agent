from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from shopping_agent.agent_graph import ShoppingAgentGraph
from shopping_agent.agent_service import AgentService
from shopping_agent.agent_store import AgentStore
from shopping_agent.catalog import SQLiteCatalog
from shopping_agent.config import Settings
from shopping_agent.demo_data import load_demo_data
from shopping_agent.embeddings import CachedEmbedder, Embedder, HashingEmbedder, LMStudioEmbedder
from shopping_agent.llm import LMStudioClient, StructuredLLM, UnavailableLLMClient
from shopping_agent.planning import Planner
from shopping_agent.retrieval import HybridSearchEngine
from shopping_agent.tools import ShoppingTools
from shopping_agent.tracing import LangfuseBridge
from shopping_agent.vector_index import InMemoryVectorIndex, QdrantVectorIndex, VectorIndex


@dataclass
class Runtime:
    catalog: SQLiteCatalog
    search_engine: HybridSearchEngine
    tools: ShoppingTools
    agent_store: AgentStore
    agent_service: AgentService
    indexed_products: int
    index_warning: str | None = None


def build_runtime(
    settings: Settings,
    *,
    embedder: Embedder | None = None,
    vector_index: VectorIndex | None = None,
    llm_client: StructuredLLM | None = None,
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
    tools = ShoppingTools(catalog, search_engine)
    agent_store = AgentStore(settings.catalog_db_path)
    if llm_client is None:
        llm_client = LMStudioClient(
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            timeout_seconds=settings.llm_timeout_seconds,
            max_output_tokens=settings.llm_max_output_tokens,
            max_retries=settings.llm_max_retries,
            store=agent_store,
        )
    observability = LangfuseBridge(
        settings.langfuse_public_key,
        settings.langfuse_secret_key,
        settings.langfuse_host,
    )
    planner = Planner(catalog, llm_client, settings.prompt_path)
    agent_graph = ShoppingAgentGraph(
        planner=planner,
        tools=tools,
        catalog=catalog,
        settings=settings,
        observability=observability,
    )
    agent_service = AgentService(
        graph=agent_graph,
        store=agent_store,
        settings=settings,
        observability=observability,
    )
    return Runtime(
        catalog=catalog,
        search_engine=search_engine,
        tools=tools,
        agent_store=agent_store,
        agent_service=agent_service,
        indexed_products=indexed,
        index_warning=warning,
    )


def build_test_runtime(settings: Settings, demo_data_dir: Path) -> Runtime:
    return build_runtime(
        settings,
        embedder=HashingEmbedder(),
        vector_index=InMemoryVectorIndex(),
        llm_client=UnavailableLLMClient(),
        demo_data_dir=demo_data_dir,
    )
