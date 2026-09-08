from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Protocol

import httpx

from shopping_agent.catalog import SQLiteCatalog

TOKEN_PATTERN = re.compile(r"[\w]+", re.UNICODE)


class Embedder(Protocol):
    model_name: str

    def embed(self, texts: Sequence[str], *, kind: str) -> list[list[float]]: ...


class LMStudioEmbedder:
    def __init__(self, base_url: str, model_name: str, timeout_seconds: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds

    def embed(self, texts: Sequence[str], *, kind: str) -> list[list[float]]:
        if not texts:
            return []
        # Nomic recommends distinct prefixes for documents and search queries.
        prefix = "search_query: " if kind == "query" else "search_document: "
        payload = {"model": self.model_name, "input": [prefix + text for text in texts]}
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(f"{self.base_url}/embeddings", json=payload)
            response.raise_for_status()
        data = sorted(response.json()["data"], key=lambda item: item["index"])
        return [item["embedding"] for item in data]


class CachedEmbedder:
    def __init__(self, inner: Embedder, catalog: SQLiteCatalog) -> None:
        self.inner = inner
        self.catalog = catalog
        self.model_name = inner.model_name

    def embed(self, texts: Sequence[str], *, kind: str) -> list[list[float]]:
        result: list[list[float] | None] = [None] * len(texts)
        missing_indexes: list[int] = []
        missing_texts: list[str] = []
        keys: list[str] = []
        for index, text in enumerate(texts):
            cache_key = hashlib.sha256(f"{kind}\0{text}".encode()).hexdigest()
            keys.append(cache_key)
            cached = self.catalog.get_cached_embedding(cache_key, self.model_name)
            if cached is None:
                missing_indexes.append(index)
                missing_texts.append(text)
            else:
                result[index] = cached

        if missing_texts:
            vectors = self.inner.embed(missing_texts, kind=kind)
            for index, vector in zip(missing_indexes, vectors, strict=True):
                result[index] = vector
                self.catalog.put_cached_embedding(keys[index], self.model_name, vector)
        return [vector for vector in result if vector is not None]


class HashingEmbedder:
    """Быстрый детерминированный эмбеддер для тестов без внешнего сервера."""

    def __init__(self, dimensions: int = 128) -> None:
        self.dimensions = dimensions
        self.model_name = f"hashing-{dimensions}"

    def embed(self, texts: Sequence[str], *, kind: str) -> list[list[float]]:
        return [self._one(text) for text in texts]

    def _one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in TOKEN_PATTERN.findall(text.casefold()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            position = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[position] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]
