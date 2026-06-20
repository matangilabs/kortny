# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) and human contributors working in this repository.

Kortny is a self-hosted, Slack-native AI coworker. Slack events become durable Postgres-backed tasks; a background worker executes each task through an LLM + tool loop and posts the result back to the originating thread. Ambient subsystems (observation, knowledge graph, witness, scheduler) run alongside the request path to give Kortny memory and proactivity.

## Commands

```bash
# Install dependencies (dev group included by default)
uv sync

# All checks: lint, format-check, typecheck, test
make check

# Individual checks
make lint          # ruff check
make lint-fix      # ruff check --fix
make format        # ruff format
make format-check  # ruff format --check
make typecheck     # mypy (strict; 0 errors is the baseline)
make test          # pytest -n auto --dist loadfile (parallel, per-worker DB clones)
make test-serial   # pytest (single process, single test DB)

# Single test (serial path)
KORTNY_TEST_POSTGRES_URL=postgresql://kortny:kortny@localhost:5432/kortny_test \
  uv run pytest tests/path/to/test_file.py::test_name

# CI-equivalent local run (mimics .github/workflows/ci.yml; no .env)
KORTNY_TEST_POSTGRES_URL=postgresql://kortny:kortny@localhost:5432/kortny_test \
  ENCRYPTION_KEY=ci-only-test-key COMPOSIO_API_KEY=ci-only-test-key \
  uv run pytest -q -n auto --dist loadfile

# Database migrations
make migrate       # alembic upgrade head
make downgrade     # alembic downgrade base

# Run services (Docker Compose) — brings up all default services
make compose-up                    # postgres, migrate, app, worker, scheduler, witness, dashboard, sandbox
make compose-up-observability      # + Phoenix OTEL tracing (observability profile)
docker compose --profile temporal up -d   # + Temporal engine and worker (temporal profile)
make compose-logs-workflow         # app + worker + temporal + temporal-worker logs
```

## Architecture

### Request flow

1. **`kortny/slack/`** — Slack Bolt app over **Socket Mode** (`kortny.slack.__main__`). `SlackIngress` converts incoming events (app mentions, DMs, soft channel mentions, `member_joined_channel`, reactions) into `Task` rows via `TaskService` (`kortny/tasks/service.py`). Handlers ack immediately, then do work. The intent classifier (`kortny/intent/`) decides whether a soft channel mention warrants a task and emits an `IntentDecision` (classification, model tier, likely tools). Reactions drive cancel (`x`), retry (`arrows_counterclockwise`), and approval (`white_check_mark` / `no_entry_sign`).

2. **`kortny/queue/`** — Postgres queue using `SELECT ... FOR UPDATE SKIP LOCKED`. Claiming sets `locked_by` + `lease_expires_at` (default 300s lease); expired leases are reclaimed and requeued with backoff.

3. **`kortny/worker/`** — Background worker (`kortny.worker.__main__`). `AgentTaskExecutor.execute()` runs, in order: Tier-0 deterministic fast paths (`kortny/routing/tier0.py`, currently schedule-state queries — no LLM), the channel graph-refresh pipeline for assessment tasks, then the general agent runtime. After success it runs post-completion hooks: post result to Slack, record routing trace, project witness opportunities, reinforce graph context, project task-summary graph entities, mark channel assessment complete, ack reaction.

4. **`kortny/agent/`** — Agent runtimes implement the `AgentRuntime` protocol in `runtime.py`. `CustomAgentRuntime` (the only runtime; `AGENT_RUNTIME=custom`) wraps the `AgentCoordinator` LLM loop in `coordinator.py`. (The Google ADK runtime was retired in HIG-281; `AGENT_RUNTIME=adk` now fails loudly.) The coordinator calls LiteLLM via `LLMService`, dispatches tool calls from `ToolRegistry`, records every turn as `TaskEvent` rows, and enforces guardrails from `execution.py`: max 6 turns, 12 tool calls, 4 recoverable failures, circuit breaker at 2 identical tool calls / 2 identical recoverable errors. Two execution modes: inline (default) and planned (`ExecutionPlanner` authors an explicit plan first; falls back to inline on planner failure).

5. **`kortny/workflow/`** — Temporal integration (optional). `KORTNY_WORKFLOW_BACKEND=temporal` routes tasks through `kortny.workflow.__main__` instead of the inline worker. `planning_classifier.py` and the semantic router (`kortny/routing/semantic.py`) are currently **observe-only/shadow** — they record decisions but do not change execution.

6. **Results** — `SlackPoster` posts `result_summary` back to the originating thread through `SlackSideEffectOutbox` (idempotency-keyed, so retries never double-post).

### Context assembly

`kortny/agent/context.py` builds the message list in this order: system prompt → acknowledgement context → known facts (workspace/channel/user scoped) → prior thread context → episodes → knowledge graph context → `<available_skills>` block → user input. Each section has a character budget (facts 4k, episodes 4k/5 max, graph 1.5k/12 items, skills 4k/30 max, thread 12k); overflows are recorded as `ContextOmission` on the `ContextPackage`.

### Ambient loops

- **`kortny/observe/`** — passive channel observation. Messages land as `ObservationEvent` rows gated by `ObservePolicy` (per workspace/channel/user: observation off/passive/active, proactivity off/digest_only/full, retention default 90 days). Channel assessments produce `ObserveChannelProfile` rows (summary, confidence, evidence) via synthetic tasks.
- **`kortny/witness/`** (`kortny.witness` service) — polls profiles every 300s (advisory-lock leader election), extracts up to 5 opportunity candidates per profile via LLM, dedupes into `WitnessOpportunityCandidate` (statuses: candidate → sent → accepted/dismissed/cooldown), and delivers rate-limited suggestions to Slack.
- **`kortny/scheduler/`** (`kortny.scheduler` service) — materializes due `Schedule` rows (oneoff/interval/cron; catchup + overlap policies) into tasks every 5s under an advisory lock. `llm_parser.py` turns natural-language schedule requests into `ScheduleDraft`s (min confidence 0.72, otherwise asks a clarifying question).

### Tool pipeline

Catalog (`kortny/tools/catalog.py`, `ToolMetadata` per tool) → per-task narrowing (`kortny/tool_selection/`, LLM selector on the cheap tier with deterministic heuristic fallback) → runtime registry (`kortny/tools/registry.py`) → approval gate (`kortny/approvals.py`). External tools arrive via the `ExternalToolProvider` seam: `kortny/composio/provider.py` and `kortny/mcp/provider.py` both emit tool cards + runtime tools, so they flow through selection and approvals identically. MCP tools are named `mcp__<server>__<tool>`; read-only hints (`readOnlyHint`) clear the approval requirement for both Composio and MCP.

Approvals: read-only native tools need none; sandbox/code/deploy tools need user approval; untrusted code paths need admin approval; approval keys hash normalized arguments so an identical retried call doesn't re-prompt. `ToolApprovalRequired` pauses the task (`waiting_approval`) until the user reacts in Slack.

### Key modules

| Module | Purpose |
|--------|---------|
| `kortny/config/settings.py` | Pydantic `Settings` — single source for all env-var-backed config. Check here before guessing any env var |
| `kortny/db/models.py` | SQLAlchemy ORM — all tables (see Data model below) |
| `kortny/tasks/` | `TaskService` / `TaskRepository` — task creation, identity + dedup, lifecycle |
| `kortny/tools/` | Tool registry + native tools; `catalog.py` holds per-tool `ToolMetadata`, `native_runtime.py` wires factories |
| `kortny/tool_selection/` | Narrows the catalog to a per-task tool set (LLM selector, budgets, providers) |
| `kortny/routing/` | Tier-0 deterministic routes + shadow semantic router + `RoutingDecisionTrace` |
| `kortny/llm/` | LiteLLM wrapper, tier routing, DB-backed provider config (`runtime_config.py`); `LLMService` records every call's usage + cost |
| `kortny/memory/` | `WorkspaceState` facts (propose → Slack confirm/reject → active; secret-pattern proposals blocked) + `Episode` per-task summaries (retrieved by same-thread → same-channel → same-user) |
| `kortny/composio/` | Composio integration provider; scoped connections, read-only verb/tag detection |
| `kortny/mcp/` | MCP client (stdio / Streamable HTTP / SSE via official `mcp` SDK), provider, dashboard-registered servers with encrypted secrets |
| `kortny/dashboard/` | FastAPI operator dashboard (port 8080); login (bootstrap/Slack OIDC/hybrid), roles admin/member |
| `kortny/observability/` | OpenTelemetry tracing (`start_span`, `set_span_attributes`); Phoenix as local OTEL backend; content capture modes metadata/summaries/full |
| `kortny/observe/` | Ambient channel observation + profile assessment (passive, no task created) |
| `kortny/witness/` | Autopilot worker — proactive opportunity extraction + delivery |
| `kortny/scheduler/` | Postgres-native schedule materializer + LLM schedule parser |
| `kortny/intent/` | LLM intent classifier for ingress (classification, tier, tool hints) |
| `kortny/agent/` | Runtimes, coordinator loop, context assembly, execution guardrails, planner |
| `kortny/knowledge_graph/` | Workspace knowledge graph: entities/edges with lifecycle + visibility scoping, evidence provenance, channel refresh pipeline, reinforcement |
| `kortny/skills/` | Agent skills: `builtins.py`, `curated/` SKILL.md catalog, `ingestion.py` (dir/zip/markdown import), trust tiers, scoped enablements; loaded at execution time via `load_skill` / `load_skill_resource` / `run_skill_script` |
| `kortny/sandbox_runner/` + `kortny/execution/` | Sandboxed code execution: HTTP runner service → docker-socket-proxy → throwaway containers; ephemeral `code_exec` and persistent per-task workbench sessions |
| `kortny/approvals.py` | Tool approval gate (none / self-gated / user / admin) |

## Data model

Schema source of truth: the SQLAlchemy models (`kortny/db/models.py`) + the Alembic migrations in `kortny/db/migrations/versions/` (`NNNN_slug.py`). `docs/schema.dbml` is a generated reference snapshot (regen command in its header) — don't hand-edit it. Always create new migrations with Alembic; **never edit applied ones**.

- **Task identity & dedup** — `Task.identity_key` is unique per installation. Kinds: `slack_message` (`slack-message:{channel}:{thread_ts}:{message_ts}`), `slack_event`, `synthetic` (observe/assessment), `scheduled` (`scheduled:{schedule_id}:{fire_time}`), `manual` (no dedup). Creating a task with an existing key returns the existing row — in tests, give each task a distinct `message_ts` or you'll silently get the same task back.
- **Task statuses** — pending → running → succeeded/failed/waiting_approval/cancelled/crashed. `TaskEvent` is the append-only audit log (llm_call, tool_call, tool_result, artifact_created, message_posted, error, log, …).
- **Knowledge graph** — entities/edges carry `lifecycle_state` (candidate → active/confirmed → stale/superseded/contradicted/archived), `visibility_scope_type` (workspace/channel/private_channel/dm/user), `source_type` provenance, and `KnowledgeGraphEvidence` rows pointing back to tasks/episodes/observations.
- **Secrets at rest** — MCP server secrets and provider API keys are Fernet-encrypted with `ENCRYPTION_KEY` (any string works; a key is derived).

## Environment variables

Copy `.env.example` to `.env`. All vars are declared in `kortny/config/settings.py` — that file is authoritative.

**Required** (Settings will not load without them): `SLACK_BOT_TOKEN`, `SLACK_APP_TOKEN`, `SLACK_SIGNING_SECRET`, `LLM_PROVIDER` (`openai`|`anthropic`|`openrouter`), `LLM_API_KEY`, `LLM_MODEL`, `POSTGRES_URL`, `COMPOSIO_API_KEY`.

**Model tiers** (optional, fall back along a chain to `LLM_MODEL`): `LLM_CHEAP_MODEL`, `LLM_STANDARD_MODEL`, `LLM_ANALYSIS_MODEL`, `LLM_DOCUMENT_MODEL`, `LLM_HIGH_REASONING_MODEL`, `LLM_HUMANIZER_MODEL`, `LLM_VISION_MODEL`. Tasks with image uploads are routed deterministically to `LLM_VISION_MODEL` (before any intent classification); it falls back to `LLM_MODEL`, and if neither is vision-capable the LLMService fail-loud check fires (HIG-279). The dashboard model-config pages can override these from the DB (`kortny/llm/runtime_config.py`); env is the fallback. Note: LiteLLM runtime identifiers are provider-prefixed (e.g. `openrouter/anthropic/...`) by `litellm_catalog.py`.

**Feature flags & integrations**: `KORTNY_WORKFLOW_BACKEND` (`inline` default / `temporal`), `KORTNY_MCP_ENABLED`, `KORTNY_WITNESS_ENABLED`, `RESPONSE_HUMANIZER_ENABLED`, `OBSERVABILITY_ENABLED` + `OTEL_EXPORTER_OTLP_ENDPOINT` (Phoenix: `http://phoenix:6006/v1/traces`), `BRAVE_SEARCH_API_KEY` (web search), `ENCRYPTION_KEY` (secrets at rest), `KORTNY_SANDBOX_RUNNER_URL` (unset disables code execution), `KORTNY_SCHEDULER_POLL_INTERVAL_SECONDS` / `KORTNY_SCHEDULER_MATERIALIZE_LIMIT`.

Beware: pydantic-settings fills any field you don't pass explicitly from `.env`. Tests constructing `Settings` must pass every field they assert on, or local runs will diverge from CI (which has no `.env`).

## Services in compose.yaml

- `postgres` — primary store (tasks, events, memory, all state)
- `migrate` — runs `alembic upgrade head` before app/worker boot
- `app` — Slack Bolt event handler (`kortny.slack`)
- `worker` — task executor (`kortny.worker`)
- `ambient` — merged poller service (`kortny.ambient`): hosts the scheduler materializer, witness runner, and consolidator worker as supervised threads in one process, with per-loop crash isolation + exponential restart backoff (HIG-234). Replaces the former `scheduler` / `witness` / `consolidator` services. Advisory locks make multiple instances safe, so the split entrypoints (`python -m kortny.scheduler` / `kortny.witness` / `kortny.consolidator`) still run individually as the scale-out path — peel any loop back into its own container with no double-work.
- `dashboard` — FastAPI operator UI (`kortny.dashboard.app`, port 8080)
- `sandbox-docker-proxy` + `sandbox-runner` — code-execution sandbox (runner on 8090; containers reach Docker only through the socket proxy)
- `temporal` + `temporal-worker` — optional durable workflow engine (profile: `temporal`)
- `phoenix` — optional OTEL/tracing UI (profile: `observability`)

Dashboard and app containers are volume-mounted but run **without** auto-reload — `docker compose restart dashboard` (or `app`/`worker`) after Python changes.

## Testing

Tests use `pytest-asyncio` (`asyncio_mode = "auto"`) against **real Postgres** — integration tests rely on actual query behavior (SKIP LOCKED, JSONB, advisory locks). Never mock the DB.

- `make test` runs parallel: `tests/conftest.py` migrates `KORTNY_TEST_POSTGRES_URL` once as a template, clones one DB per xdist worker (`kortny_test_gw0`, …) via `CREATE DATABASE ... TEMPLATE`, and each worker rewrites the env var before test modules import. `--dist loadfile` keeps each file on one worker.
- Serial runs (`make test-serial`, single-file debugging) share the one base test DB — never run two serial pytest processes at once (DELETE-cleanup fixtures deadlock behind idle-in-transaction sessions). One suite run at a time either way.
- The test DB name must end in `_test`, start with `test_`, or contain `_test_` (`tests/db_safety.py` refuses anything else, and refuses any target matching `POSTGRES_URL`).
- DB-backed test files skip themselves when `KORTNY_TEST_POSTGRES_URL` is unset.
- To reproduce CI failures locally, use the CI-equivalent command in the Commands section (no `.env`, dummy `ENCRYPTION_KEY`/`COMPOSIO_API_KEY`).
- Debug hangs with `pytest -o faulthandler_timeout=120` and `pg_stat_activity`. Don't pipe pytest through `tail`/`grep` when the exit code matters — pipes mask it.

## CI

`.github/workflows/ci.yml` runs on pushes to `main` and all PRs — two parallel jobs:
- **lint**: ruff check, ruff format --check, mypy (with `.mypy_cache` restored via actions/cache).
- **test**: Postgres 16 service container, alembic migrate, `pytest -n auto --dist loadfile`.

mypy is strict and the baseline is 0 errors in all files — don't introduce `# type: ignore` without a specific error code and a reason. Run `make check` before pushing.

## Conventions

### Tool authoring

New tools implement the `Tool` protocol in `kortny/tools/types.py` (`name`, `description`, `parameters` JSON schema, `invoke(args) -> ToolResult`). Register a factory in `kortny/tools/native_runtime.py` **and** a `ToolMetadata` entry in `kortny/tools/catalog.py` (namespace, category, capabilities, `side_effect` read/write/destructive, `approval` none/self_gated/user_approval/admin_approval, required env vars/Slack scopes) — add both or the tool won't surface. Return `ToolResult`; raise `RecoverableToolError(code, message, hint)` for errors the coordinator should retry or route around. Destructive tools should declare a `ToolSandboxPolicy`.

### Dashboard

Server-rendered Jinja2 + a single hand-rolled CSS system (`kortny/dashboard/static/dashboard.css`) with light/dark themes. Use the existing design tokens (`--card`, `--border`, `--muted`, `--radius`) and component classes: `.btn` / `.btn-ghost` (there is **no** `.btn-primary`), `.badge badge-{success|warning|danger|neutral|accent}`, `.card`, segmented-nav. Action handlers follow the pattern: `require_admin` dependency → `parse_qs(body)` → service call → `_redirect_with_notice(...)` with `_safe_next_path`, auditing via `dashboard_actor(...)`. Read-only pages take `require_principal`; anything mutating takes `require_admin`.

### Skills

A skill is a directory with `SKILL.md` (frontmatter: name, description, tags) plus optional `references/` and `scripts/`. Curated skills live in `kortny/skills/curated/` and are seeded on startup; custom skills are ingested from the dashboard (directory zip or pasted markdown) and default to `untrusted`. Trust ladder: `trusted` → `community` → `untrusted` → `quarantined`; scripts execute only at `trusted`, and only inside the sandbox. Enablement is scoped (workspace/channel/user; narrowest wins). The agent sees an `<available_skills>` index and loads content progressively (`load_skill` → `load_skill_resource` → `run_skill_script`).

### Code style

- Python 3.11, line length per `ruff` config; `ruff format` is the formatter — run it, don't hand-format.
- mypy strict: annotate everything; prefer real types/`cast()` over ignores; Protocol implementations must match parameter types exactly (use `Sequence[...]`, not concrete tuples).
- Settings objects in tests: build via `Settings.model_validate({...})` with UPPERCASE alias keys.
- User-facing Slack text is normalized at the posting boundary (`normalize_user_facing_text`, `normalize_slack_mrkdwn` in `kortny/slack_mrkdwn.py`) — don't hand-sanitize copy in feature code.
- LLM calls go through `LLMService` so usage and cost are recorded — never call LiteLLM directly from feature code.
