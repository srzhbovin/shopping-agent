from scripts.import_amazon_reviews import metadata_to_product, raw_to_review


def test_amazon_metadata_mapping_keeps_catalog_id() -> None:
    product = metadata_to_product(
        {
            "parent_asin": "B012345678",
            "title": "Steel pan",
            "price": "$29.95",
            "average_rating": 4.5,
            "rating_number": 42,
            "features": ["No glass", "26 cm"],
            "details": {"Material": "Steel"},
        },
        "Home and Kitchen",
    )

    assert product is not None
    assert product.product_id == "B012345678"
    assert str(product.price) == "29.95"
    assert product.source_url == "https://www.amazon.com/dp/B012345678"


def test_amazon_review_mapping_rejects_product_outside_loaded_subset() -> None:
    review = raw_to_review(
        {
            "parent_asin": "NOT-LOADED",
            "rating": 5,
            "text": "Good product",
            "user_id": "U1",
            "timestamp": 1,
        },
        {"B012345678"},
    )

    assert review is None
