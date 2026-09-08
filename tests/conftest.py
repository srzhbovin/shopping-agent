from pathlib import Path

import pytest

from shopping_agent.config import Settings
from shopping_agent.runtime import Runtime, build_test_runtime


@pytest.fixture()
def demo_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "demo"


@pytest.fixture()
def runtime(tmp_path: Path, demo_dir: Path) -> Runtime:
    settings = Settings(
        catalog_db_path=tmp_path / "catalog.db",
        auto_load_demo=True,
    )
    return build_test_runtime(settings, demo_dir)
