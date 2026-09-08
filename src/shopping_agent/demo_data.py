from __future__ import annotations

import json
from pathlib import Path

from shopping_agent.catalog import SQLiteCatalog
from shopping_agent.models import Product, Review


def load_demo_data(catalog: SQLiteCatalog, data_dir: Path) -> tuple[int, int]:
    products_raw = json.loads((data_dir / "products.json").read_text(encoding="utf-8"))
    reviews_raw = json.loads((data_dir / "reviews.json").read_text(encoding="utf-8"))
    return (
        catalog.upsert_products(Product.model_validate(item) for item in products_raw),
        catalog.upsert_reviews(Review.model_validate(item) for item in reviews_raw),
    )
