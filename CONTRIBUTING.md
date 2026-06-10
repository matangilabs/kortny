# Contributing to Kortny

Thanks for your interest in making Kortny better. Contributions of all
kinds are welcome: bug reports, docs fixes, new native tools, and bigger
features.

## Quick start

```sh
# 1. Fork and clone, then install dependencies
uv sync

# 2. Copy environment config (see README for Slack/LLM setup)
cp .env.example .env

# 3. Run the full check suite (lint, format, typecheck, tests)
make check

# 4. Bring up the stack
make compose-up
```

### Running tests

```sh
make test                # unit tests (no database needed)

# DB-backed integration tests need a dedicated test database.
# NEVER point this at your dev database — tests truncate tables.
KORTNY_TEST_POSTGRES_URL=postgresql://kortny:kortny@localhost:5432/kortny_test uv run pytest
```

Run a single test with:

```sh
uv run pytest tests/path/to/test_file.py::test_name
```

## Making changes

- Create a feature branch off `main`.
- Keep PRs focused: one fix or feature per PR.
- `make check` must pass (ruff lint + format, mypy, pytest).
- Add or update tests for any behavior change. Tests live in `tests/`
  and use `pytest-asyncio` with `asyncio_mode = "auto"`.
- Database changes go through Alembic migrations in
  `kortny/db/migrations/versions/` — never edit an applied migration.
- New tools implement the `Tool` interface in `kortny/tools/types.py`
  and need both a factory registration in
  `kortny/tools/native_runtime.py` and a `ToolMetadata` entry in
  `kortny/tools/catalog.py`. See the "Tool authoring" section of
  `CLAUDE.md` for details.

## Commit messages

Use conventional-commit style prefixes (`feat:`, `fix:`, `docs:`,
`refactor:`, `test:`, `build:`) with a scope where it helps, e.g.
`feat(sandbox): add session reaper`.

## Developer Certificate of Origin

By contributing, you certify the [Developer Certificate of
Origin](https://developercertificate.org/) — that you wrote the
contribution or otherwise have the right to submit it under the
project's [Apache-2.0 license](./LICENSE). Sign off your commits with
`git commit -s` (adds a `Signed-off-by` line).

## Reporting bugs and proposing features

- **Bugs:** open a GitHub issue with reproduction steps, expected vs.
  actual behavior, and relevant logs (redact tokens and workspace data).
- **Security issues:** do **not** open a public issue — see
  [SECURITY.md](./SECURITY.md).
- **Features:** open an issue describing the problem first. Kortny
  favors small, composable tools and harness-owned guardrails over
  model-owned behavior; proposals aligned with that direction land
  fastest.
