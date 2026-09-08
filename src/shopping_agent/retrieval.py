from __future__ import annotations

import math
import re
from collections import Counter

from shopping_agent.catalog import SQLiteCatalog
from shopping_agent.embeddings import Embedder
from shopping_agent.models import Product, SearchCatalogInput, SearchCatalogOutput, SearchHit
from shopping_agent.vector_index import VectorIndex

TOKEN_PATTERN = re.compile(r"[\w]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [token for token in TOKEN_PATTERN.findall(text.casefold()) if len(token) > 1]


class BM25Index:
    def __init__(self, products: list[Product], k1: float = 1.5, b: float = 0.75) -> None:
        self.products = products
        self.k1 = k1
        self.b = b
        self.tokens = {
            product.product_id: tokenize(product.searchable_text()) for product in products
        }
        self.lengths = {key: len(value) for key, value in self.tokens.items()}
        self.average_length = sum(self.lengths.values()) / max(len(products), 1)
        self.document_frequency: Counter[str] = Counter()
        for tokens in self.tokens.values():
            self.document_frequency.update(set(tokens))

    def search(self, query: str, allowed_ids: set[str], limit: int) -> list[tuple[str, float]]:
        query_tokens = tokenize(query)
        document_count = max(len(self.products), 1)
        scored: list[tuple[str, float]] = []
        for product_id, document_tokens in self.tokens.items():
            if product_id not in allowed_ids:
                continue
            frequencies = Counter(document_tokens)
            score = 0.0
            for token in query_tokens:
                frequency = frequencies[token]
                if frequency == 0:
                    continue
                df = self.document_frequency[token]
                idf = math.log(1 + (document_count - df + 0.5) / (df + 0.5))
                normalization = frequency + self.k1 * (
                    1 - self.b + self.b * self.lengths[product_id] / max(self.average_length, 1)
                )
                score += idf * (frequency * (self.k1 + 1)) / normalization
            if score > 0:
                scored.append((product_id, score))
        return sorted(scored, key=lambda item: (-item[1], item[0]))[:limit]


class HybridSearchEngine:
    def __init__(
        self,
        catalog: SQLiteCatalog,
        embedder: Embedder,
        vector_index: VectorIndex,
        rrf_k: int = 60,
    ) -> None:
        self.catalog = catalog
        self.embedder = embedder
        self.vector_index = vector_index
        self.rrf_k = rrf_k
        self._products: list[Product] = []
        self._bm25 = BM25Index([])

    def rebuild(self) -> int:
        self._products = self.catalog.list_products()
        self._bm25 = BM25Index(self._products)
        if not self._products:
            return 0
        vectors = self.embedder.embed(
            [product.searchable_text() for product in self._products], kind="document"
        )
        self.vector_index.replace(
            [
                (product.product_id, vector)
                for product, vector in zip(self._products, vectors, strict=True)
            ]
        )
        return len(self._products)

    def search(self, request: SearchCatalogInput) -> SearchCatalogOutput:
        candidates = self.catalog.list_products(request.filters)
        allowed_ids = {product.product_id for product in candidates}
        product_by_id = {product.product_id: product for product in candidates}
        candidate_limit = min(max(request.top_k * 5, 30), 200)
        lexical = self._bm25.search(request.query, allowed_ids, candidate_limit)
        warnings: list[str] = []
        dense: list[tuple[str, float]] = []
        try:
            query_vector = self.embedder.embed([request.query], kind="query")[0]
            dense = [
                item
                for item in self.vector_index.search(query_vector, candidate_limit)
                if item[0] in allowed_ids
            ]
        except Exception as error:  # lexical fallback is an intentional degradation path
            warnings.append(f"Dense retrieval unavailable, lexical fallback used: {error}")

        lexical_rank = {product_id: rank for rank, (product_id, _) in enumerate(lexical, start=1)}
        dense_rank = {product_id: rank for rank, (product_id, _) in enumerate(dense, start=1)}
        all_ids = set(lexical_rank) | set(dense_rank)
        if not all_ids and not tokenize(request.query):
            all_ids = allowed_ids

        query_terms = set(tokenize(request.query))
        hits: list[SearchHit] = []
        for product_id in all_ids:
            score = 0.0
            if product_id in lexical_rank:
                score += 1.0 / (self.rrf_k + lexical_rank[product_id])
            if product_id in dense_rank:
                score += 0.85 / (self.rrf_k + dense_rank[product_id])
            product = product_by_id[product_id]
            matched_terms = sorted(query_terms & set(tokenize(product.searchable_text())))
            # A transparent, deterministic reranking signal. A learned reranker is added in stage 3.
            score += 0.002 * len(matched_terms)
            if product.rating is not None:
                score += 0.0001 * product.rating
            hits.append(
                SearchHit(
                    product=product,
                    score=score,
                    lexical_rank=lexical_rank.get(product_id),
                    dense_rank=dense_rank.get(product_id),
                    matched_terms=matched_terms,
                )
            )
        hits.sort(key=lambda hit: (-hit.score, hit.product.product_id))
        mode = "hybrid" if dense else "bm25"
        return SearchCatalogOutput(
            query=request.query,
            hits=hits[: request.top_k],
            retrieval_mode=mode,
            warnings=warnings,
        )
