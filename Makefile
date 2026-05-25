.PHONY: install lock lint lint-fix format format-check typecheck test check migrate downgrade compose-up compose-up-observability compose-down compose-down-observability compose-down-observability-volumes compose-logs compose-logs-observability playground clean

install:
	uv sync

lock:
	uv lock

lint:
	uv run ruff check .

lint-fix:
	uv run ruff check . --fix

format:
	uv run ruff format .

format-check:
	uv run ruff format --check .

typecheck:
	uv run mypy .

test:
	uv run pytest

check: lint format-check typecheck test

migrate:
	uv run alembic upgrade head

downgrade:
	uv run alembic downgrade base

compose-up:
	docker compose up

compose-up-observability:
	OTEL_EXPORTER_OTLP_ENDPOINT=http://phoenix:6006/v1/traces docker compose --profile observability up -d

compose-down:
	docker compose down

compose-down-observability:
	docker compose --profile observability down

compose-down-observability-volumes:
	docker compose --profile observability down -v

compose-logs:
	docker compose logs -f

compose-logs-observability:
	docker compose --profile observability logs -f app worker phoenix

playground:
	uv run adk web .

clean:
	rm -rf .mypy_cache .pytest_cache .ruff_cache htmlcov .coverage build dist
