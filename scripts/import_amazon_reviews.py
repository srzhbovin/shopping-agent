from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, TextIO

from shopping_agent.catalog import SQLiteCatalog
from shopping_agent.models import Product, Review


def open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def json_lines(path: Path) -> Iterator[dict[str, Any]]:
    with open_text(path) as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def clean_price(value: Any) -> str | None:
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        cleaned = "".join(
            character for character in value if character.isdigit() or character == "."
        )
        return cleaned or None
    return None


def joined(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value or "")


def metadata_to_product(raw: dict[str, Any], category: str) -> Product | None:
    product_id = raw.get("parent_asin") or raw.get("asin")
    price = clean_price(raw.get("price"))
    title = str(raw.get("title") or "").strip()
    if not product_id or not price or not title:
        return None
    details = raw.get("details") if isinstance(raw.get("details"), dict) else {}
    attributes = {str(key): str(value) for key, value in details.items() if value is not None}
    description = " ".join((joined(raw.get("description")), joined(raw.get("features")))).strip()
    return Product(
        product_id=str(product_id),
        title=title,
        category=category,
        description=description,
        price=price,
        currency="USD",
        rating=raw.get("average_rating"),
        review_count=int(raw.get("rating_number") or 0),
        attributes=attributes,
        source_url=f"https://www.amazon.com/dp/{product_id}",
    )


def raw_to_review(raw: dict[str, Any], allowed_ids: set[str]) -> Review | None:
    product_id = str(raw.get("parent_asin") or raw.get("asin") or "")
    text = str(raw.get("text") or "").strip()
    if product_id not in allowed_ids or not text:
        return None
    fingerprint = "\0".join(
        (product_id, str(raw.get("user_id") or ""), str(raw.get("timestamp") or ""), text)
    )
    review_id = hashlib.sha1(fingerprint.encode("utf-8"), usedforsecurity=False).hexdigest()
    return Review(
        review_id=review_id,
        product_id=product_id,
        rating=float(raw.get("rating") or 0),
        title=str(raw.get("title") or ""),
        text=text,
        verified_purchase=raw.get("verified_purchase"),
    )


def import_data(args: argparse.Namespace) -> None:
    catalog = SQLiteCatalog(args.db)
    products: list[Product] = []
    for raw in json_lines(args.metadata):
        product = metadata_to_product(raw, args.category)
        if product:
            products.append(product)
        if len(products) >= args.limit_products:
            break
    catalog.upsert_products(products)
    allowed_ids = {product.product_id for product in products}

    reviews: list[Review] = []
    if args.reviews:
        for raw in json_lines(args.reviews):
            review = raw_to_review(raw, allowed_ids)
            if review:
                reviews.append(review)
            if len(reviews) >= args.limit_reviews:
                break
        catalog.upsert_reviews(reviews)
    print(json.dumps({"products": len(products), "reviews": len(reviews)}, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Потоковая загрузка Amazon Reviews 2023")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--db", type=Path, default=Path("data/catalog.db"))
    parser.add_argument("--category", default="Home and Kitchen")
    parser.add_argument("--limit-products", type=int, default=10000)
    parser.add_argument("--limit-reviews", type=int, default=50000)
    return parser.parse_args()


if __name__ == "__main__":
    import_data(parse_args())
