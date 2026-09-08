.PHONY: install lint test run up down verify check-models clean-data

install:
	python -m pip install uv==0.12.10
	python -m uv sync --frozen --extra dev

test:
	python -m uv run --extra dev pytest

lint:
	python -m uv run --extra dev ruff check src tests scripts
	python -m uv run --extra dev ruff format --check src tests scripts

run:
	python -m uv run uvicorn shopping_agent.api:app --reload

up:
	docker compose up --build -d

down:
	docker compose down

verify:
	python -m uv run shopping-agent verify

check-models:
	python -m uv run shopping-agent check-models

clean-data:
	python -m uv run shopping-agent reset-demo
