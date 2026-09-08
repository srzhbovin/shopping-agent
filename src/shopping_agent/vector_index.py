from __future__ import annotations

import math
import uuid
from collections.abc import Sequence
from typing import Protocol

from qdrant_client import QdrantClient, models


class VectorIndex(Protocol):
    def replace(self, items: Sequence[tuple[str, list[float]]]) -> None: ...

    def search(self, vector: list[float], limit: int) -> list[tuple[str, float]]: ...

    def ready(self) -> bool: ...


class QdrantVectorIndex:
    def __init__(self, url: str, collection: str) -> None:
        self.client = QdrantClient(url=url, timeout=15)
        self.collection = collection

    @staticmethod
    def _point_id(product_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"catalog-product:{product_id}"))

    def replace(self, items: Sequence[tuple[str, list[float]]]) -> None:
        if not items:
            return
        dimensions = len(items[0][1])
        collections = {item.name for item in self.client.get_collections().collections}
        if self.collection in collections:
            info = self.client.get_collection(self.collection)
            current_size = info.config.params.vectors.size
            if current_size != dimensions:
                self.client.delete_collection(self.collection)
                collections.remove(self.collection)
        if self.collection not in collections:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=dimensions, distance=models.Distance.COSINE
                ),
            )
        points = [
            models.PointStruct(
                id=self._point_id(product_id),
                vector=vector,
                payload={"product_id": product_id},
            )
            for product_id, vector in items
        ]
        self.client.upsert(collection_name=self.collection, points=points, wait=True)

    def search(self, vector: list[float], limit: int) -> list[tuple[str, float]]:
        points = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=limit,
            with_payload=True,
        ).points
        return [
            (str(point.payload["product_id"]), float(point.score))
            for point in points
            if point.payload and "product_id" in point.payload
        ]

    def ready(self) -> bool:
        return self.client.collection_exists(self.collection)


class InMemoryVectorIndex:
    def __init__(self) -> None:
        self.items: dict[str, list[float]] = {}

    def replace(self, items: Sequence[tuple[str, list[float]]]) -> None:
        self.items = dict(items)

    def search(self, vector: list[float], limit: int) -> list[tuple[str, float]]:
        scored = [
            (product_id, cosine(vector, candidate)) for product_id, candidate in self.items.items()
        ]
        return sorted(scored, key=lambda item: (-item[1], item[0]))[:limit]

    def ready(self) -> bool:
        return bool(self.items)


def cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0
