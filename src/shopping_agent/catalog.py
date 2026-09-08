from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path

from shopping_agent.errors import ProductNotFoundError
from shopping_agent.models import Product, Review, SearchFilters


class SQLiteCatalog:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS products (
                    product_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    category TEXT NOT NULL,
                    description TEXT NOT NULL,
                    price TEXT NOT NULL,
                    currency TEXT NOT NULL,
                    rating REAL,
                    review_count INTEGER NOT NULL,
                    attributes_json TEXT NOT NULL,
                    source_url TEXT
                );
                CREATE TABLE IF NOT EXISTS reviews (
                    review_id TEXT PRIMARY KEY,
                    product_id TEXT NOT NULL REFERENCES products(product_id),
                    rating REAL NOT NULL,
                    title TEXT NOT NULL,
                    text TEXT NOT NULL,
                    verified_purchase INTEGER
                );
                CREATE INDEX IF NOT EXISTS idx_products_category ON products(category);
                CREATE INDEX IF NOT EXISTS idx_reviews_product ON reviews(product_id);
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    cache_key TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    vector_json TEXT NOT NULL
                );
                """
            )

    def count_products(self) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM products").fetchone()[0])

    def upsert_products(self, products: Iterable[Product]) -> int:
        rows = [
            (
                product.product_id,
                product.title,
                product.category,
                product.description,
                str(product.price),
                product.currency,
                product.rating,
                product.review_count,
                json.dumps(product.attributes, ensure_ascii=False, sort_keys=True),
                product.source_url,
            )
            for product in products
        ]
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(product_id) DO UPDATE SET
                  title=excluded.title,
                  category=excluded.category,
                  description=excluded.description,
                  price=excluded.price,
                  currency=excluded.currency,
                  rating=excluded.rating,
                  review_count=excluded.review_count,
                  attributes_json=excluded.attributes_json,
                  source_url=excluded.source_url
                """,
                rows,
            )
        return len(rows)

    def upsert_reviews(self, reviews: Iterable[Review]) -> int:
        rows = [
            (
                review.review_id,
                review.product_id,
                review.rating,
                review.title,
                review.text,
                None if review.verified_purchase is None else int(review.verified_purchase),
            )
            for review in reviews
        ]
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO reviews VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(review_id) DO UPDATE SET
                  product_id=excluded.product_id,
                  rating=excluded.rating,
                  title=excluded.title,
                  text=excluded.text,
                  verified_purchase=excluded.verified_purchase
                """,
                rows,
            )
        return len(rows)

    def get_product(self, product_id: str) -> Product:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM products WHERE product_id = ?", (product_id,)
            ).fetchone()
        if row is None:
            raise ProductNotFoundError(product_id)
        return self._row_to_product(row)

    def get_reviews(self, product_id: str) -> list[Review]:
        self.get_product(product_id)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM reviews WHERE product_id = ? ORDER BY review_id", (product_id,)
            ).fetchall()
        return [
            Review(
                review_id=row["review_id"],
                product_id=row["product_id"],
                rating=row["rating"],
                title=row["title"],
                text=row["text"],
                verified_purchase=(
                    None if row["verified_purchase"] is None else bool(row["verified_purchase"])
                ),
            )
            for row in rows
        ]

    def list_products(self, filters: SearchFilters | None = None) -> list[Product]:
        products = self._all_products()
        if filters is None:
            return products
        return [product for product in products if product_matches_filters(product, filters)]

    def _all_products(self) -> list[Product]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM products ORDER BY product_id").fetchall()
        return [self._row_to_product(row) for row in rows]

    @staticmethod
    def _row_to_product(row: sqlite3.Row) -> Product:
        return Product(
            product_id=row["product_id"],
            title=row["title"],
            category=row["category"],
            description=row["description"],
            price=Decimal(row["price"]),
            currency=row["currency"],
            rating=row["rating"],
            review_count=row["review_count"],
            attributes=json.loads(row["attributes_json"]),
            source_url=row["source_url"],
        )

    def get_cached_embedding(self, cache_key: str, model: str) -> list[float] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT vector_json FROM embedding_cache WHERE cache_key = ? AND model = ?",
                (cache_key, model),
            ).fetchone()
        return None if row is None else json.loads(row["vector_json"])

    def put_cached_embedding(self, cache_key: str, model: str, vector: list[float]) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO embedding_cache VALUES (?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                  model=excluded.model,
                  vector_json=excluded.vector_json
                """,
                (cache_key, model, json.dumps(vector)),
            )


def product_matches_filters(product: Product, filters: SearchFilters) -> bool:
    if filters.min_price is not None and product.price < filters.min_price:
        return False
    if filters.max_price is not None and product.price > filters.max_price:
        return False
    if filters.categories and product.category.casefold() not in {
        category.casefold() for category in filters.categories
    }:
        return False
    if filters.min_rating is not None:
        if product.rating is None or product.rating < filters.min_rating:
            return False
    if filters.currency and product.currency != filters.currency.upper():
        return False
    normalized_attributes = {
        key.casefold(): value.casefold() for key, value in product.attributes.items()
    }
    for key, value in filters.required_attributes.items():
        if normalized_attributes.get(key.casefold()) != value.casefold():
            return False
    searchable = product.searchable_text().casefold()
    if any(term.casefold() in searchable for term in filters.excluded_terms):
        return False
    return True
