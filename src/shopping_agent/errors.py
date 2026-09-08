class CatalogError(Exception):
    """Базовая ошибка каталога."""


class ProductNotFoundError(CatalogError):
    def __init__(self, product_id: str) -> None:
        super().__init__(f"Product {product_id!r} was not found in the catalog")
        self.product_id = product_id


class CurrencyMismatchError(CatalogError):
    pass


class BudgetExceededError(CatalogError):
    pass
