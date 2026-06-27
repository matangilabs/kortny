.PHONY: install lock lint lint-fix format format-check typecheck test test-serial check migrate downgrade compose-up compose-up-observability compose-up-workflow compose-down compose-down-observability compose-down-workflow compose-down-observability-volumes compose-logs compose-logs-observability compose-logs-workflow clean seed-sim clean-sim status-sim eval eval-smoke

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
	uv run pytest -n auto --dist loadfile

test-serial:
	uv run pytest

check: lint format-check typecheck test

migrate:
	uv run alembic upgrade head

downgrade:
	uv run alembic downgrade base

compose-up:
	docker compose up -d

compose-up-observability:
	OTEL_EXPORTER_OTLP_ENDPOINT=http://phoenix:6006/v1/traces docker compose --profile observability up -d

compose-up-workflow:
	docker compose up -d

compose-down:
	docker compose down

compose-down-observability:
	docker compose --profile observability down

compose-down-workflow:
	docker compose down

compose-down-observability-volumes:
	docker compose --profile observability down -v

compose-logs:
	docker compose logs -f

compose-logs-observability:
	docker compose --profile observability logs -f app worker phoenix

compose-logs-workflow:
	docker compose logs -f app worker temporal temporal-worker

# Demo workspace seeders (scripts/demo/).
# DB seeder: runs against the live dev DB inside compose.
# Usage: make seed-sim CHANNEL=C0123456789 [DAYS=21]
seed-sim:
	@test -n "$(CHANNEL)" || { echo "CHANNEL=<slack channel id> is required, e.g. make seed-sim CHANNEL=C0123456789"; exit 1; }
	docker compose exec worker uv run python -m scripts.demo.db_seed seed --channel $(CHANNEL) --days $(or $(DAYS),21)

clean-sim:
	docker compose exec worker uv run python -m scripts.demo.db_seed clean

status-sim:
	docker compose exec worker uv run python -m scripts.demo.db_seed status

# Orchestration evals
# --------------------
# eval: live run that executes cases against the real agent and records goldens.
#   Needs: Postgres, LLM API key, connected Composio integrations.
#   Side effect: updates kortny/evals/orchestration/fixtures/smoke_goldens.json.
#   Commit the updated file after a successful run.
eval:
	uv run python -m kortny.evals.orchestration.runner

# eval-smoke: offline replay of the smoke subset — scores committed goldens.
#   Needs: nothing. No API keys, no DB, no live agent. Runs in CI / $0.
#   Fails if any smoke case result diverges from the committed fixture.
eval-smoke:
	uv run python -m kortny.evals.orchestration.replay

clean:
	rm -rf .mypy_cache .pytest_cache .ruff_cache htmlcov .coverage build dist
