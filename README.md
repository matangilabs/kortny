# Kortny

> For teams who live in Slack.

Kortny is a self-hosted AI coworker that executes real tasks,
remembers how your team works, and gets better the longer it's
there. Not a bot that answers questions — a coworker that takes
work off your plate.

---

## The Problem

Most hosted AI tools are black boxes: opaque pricing, data you
don't control, and no real visibility into what happened. Generic
Slack bots answer questions but don't *do* the work. And nothing
self-hosted combines durable task execution, workspace memory, and
cross-tool orchestration in one package you actually own and run.

---

## What Makes Kortny Different

- **It finishes what it starts** — every request runs as a tracked
  task with a full log of every step, tool call, and decision. No
  black box, no guessing what happened.
- **No surprise bills** — every task shows the model used, tokens
  consumed, and exact cost. You always know what you're spending
  and why.
- **Reads the room** — formal in #finance, casual in #general,
  silent unless mentioned in #announcements. Each channel gets the
  version of Kortny that fits.
- **Gets better the longer it's there** — remembers past work,
  learns your preferences, and stops asking questions it already
  knows the answers to.
- **Your bot, your identity** — create your own Slack app with your
  own name and avatar. Kortny runs the brain, you own the face.
- **Runs on your infrastructure** — Docker Compose, no cloud control
  plane. Your Slack data, task history, memory, and cost logs live
  in your Postgres, and you choose the LLM provider your prompts
  go to.
- **100+ integrations** — Gmail, HubSpot, GitHub, Google Drive,
  Calendar, and more via Composio OAuth. (Composio is a third-party
  broker — see [Where your data lives](#where-your-data-lives). A
  Composio-free, bring-your-own-MCP path is on the roadmap for V1.1.)
- **BYO LLM** — OpenAI, Anthropic, or OpenRouter.

---

## Where your data lives

Kortny is self-hosted, and we want to be precise about what that
means rather than waving the word around.

**Stays in your stack:** the Slack app runs under your bot token,
and every task, step log, memory record, and cost entry is stored
in your own Postgres. Nothing about your workspace's activity is
sent to us — there is no "us" in the data path. You pick the LLM
provider, so you decide where your prompts and your team's content
are processed.

**The deliberate exception — external integrations:** connecting a
tool like Gmail or HubSpot routes through **Composio**, a
third-party OAuth and tool-execution broker. We made this trade-off
on purpose: per-tool OAuth setup is the single biggest onboarding
wall for self-hosters, and Composio removes it. The cost is that
integration traffic passes through Composio rather than staying
entirely local.

If you need a fully self-contained integration plane with zero
third-party dependency, the **bring-your-own-MCP path (V1.1)** is
built for exactly that.

---

## Quickstart

### Prerequisites
- Docker and Docker Compose
- An LLM provider key (OpenAI, Anthropic, or OpenRouter)
- A Composio API key for the integration catalog and connected-account tooling

### 1. Create your Slack app

1. Go to https://api.slack.com/apps → Create New App → From Manifest
2. Paste the contents of `manifest.json` from this repo
3. Name your bot whatever you want — this is your bot, your brand
4. Upload a custom avatar if you'd like. This repo includes the Kortny icon at
   `kortny/dashboard/static/assets/kortny_icon.png`; when the dashboard is
   running, it is served at `/static/assets/kortny_icon.png`.
5. Install the app to your workspace
6. Copy your **Bot Token** (`xoxb-...`), **App-Level Token**
   (`xapp-...` with `connections:write` for Socket Mode), and
   **Signing Secret**
7. Copy the app's **Client ID** and **Client Secret** for dashboard
   Sign in with Slack
8. Add `http://localhost:8080/auth/slack/callback` as an OAuth redirect URL

If you update an existing Slack app from this repo's manifest, apply the
manifest changes in Slack and reinstall the app to the workspace so new event
subscriptions and scopes take effect.

### 2. Clone and configure

```
git clone https://github.com/boffti/kortny
cd kortny
cp .env.example .env
```

Edit `.env`:

```x
SLACK_BOT_TOKEN=xoxb-...
SLACK_APP_TOKEN=xapp-...
SLACK_SIGNING_SECRET=...
LLM_PROVIDER=openai          # openai | anthropic | openrouter
LLM_API_KEY=sk-...
LLM_MODEL=gpt-4o             # or claude-3-5-sonnet, etc.
COMPOSIO_API_KEY=...
DASHBOARD_AUTH_MODE=hybrid
DASHBOARD_SLACK_CLIENT_ID=...
DASHBOARD_SLACK_CLIENT_SECRET=...
DASHBOARD_SLACK_REDIRECT_URI=http://localhost:8080/auth/slack/callback
```

That is enough for the default Docker Compose stack. Everything else in
`.env.example` is optional and has a local-development default: bootstrap
dashboard fallback credentials, Postgres credentials, Temporal, scheduler,
Witness, and observability.

Set `BRAVE_SEARCH_API_KEY` only if you want the built-in Brave-backed web
search tool; search can also be provided by a connected Composio integration.
Set `ENCRYPTION_KEY` before saving dashboard-managed provider or integration
secrets.

### 3. Start Kortny

```
docker compose up -d --force-recreate
```

This starts Postgres on `localhost:5432`, runs the Alembic migration, starts
the Slack Socket Mode ingress service, starts the task worker, starts the
Postgres-native schedule materializer, starts the Witness proactive runner, and
serves the operator dashboard at `http://localhost:8080`. It also starts the
sandbox runner used for isolated code execution.

This does not start optional services such as Phoenix or the Temporal
experiment profile.

Witness is on by default: it creates proactive opportunity candidates, reviews
due candidates, and can start low-risk read-only proactive tasks through the
normal Kortny worker path. Autopilot is intentionally bounded: it only executes
non-interruptive read-only analysis/status checks, skips schedule-management or
confirmation-seeking follow-ups, ignores candidates produced by scheduled task
runs, and requires active channel membership before posting. The default
autopilot limit is one proactive task per tick so old candidate backlog does not
flood a workspace. It still will not send proactive DMs unless you set
`KORTNY_WITNESS_DELIVER_PRIVATE=true`.

The dashboard uses Sign in with Slack for per-user identity. In the default
`hybrid` mode, a local bootstrap login remains available for development and
recovery. Use `DASHBOARD_USERNAME` and `DASHBOARD_PASSWORD` for that fallback,
and change `DASHBOARD_SESSION_SECRET` before exposing the dashboard beyond
local development. It is bound to `127.0.0.1` by default; change
`DASHBOARD_HOST_PORT` only if you need a different local port.

If you explicitly need local-only dashboard auth, set
`DASHBOARD_AUTH_MODE=bootstrap`. Slack login stores a local dashboard user for
future personal dashboards and user-scoped integrations.

### 4. Develop against the local database

Host-side commands need a local `POSTGRES_URL`; the Compose containers receive
their internal `POSTGRES_URL` automatically:

```
export POSTGRES_URL=postgresql://kortny:kortny@localhost:5432/kortny
make migrate
docker compose exec postgres createdb -U kortny kortny_test
KORTNY_TEST_POSTGRES_URL=postgresql://kortny:kortny@localhost:5432/kortny_test uv run pytest tests/test_task_service.py tests/test_queue.py
```

Use a dedicated test database for integration tests. DB-backed tests perform
destructive cleanup and the test harness refuses to run when
`KORTNY_TEST_POSTGRES_URL` points at the default `localhost:5432/kortny`
development database. Use a database name that starts with `test_` or ends with
`_test`, such as `kortny_test`. The harness also refuses to run if
`KORTNY_TEST_POSTGRES_URL` points at the same database target as `POSTGRES_URL`,
or if `KORTNY_ENV`, `APP_ENV`, or `ENVIRONMENT` is set to `prod`/`production`.

To process at most one pending task from your host shell:

```
uv run python -m kortny.worker --once
```

To run one Witness scan/autopilot tick from your host shell:

```
uv run python -m kortny.witness --once
```

To inspect task costs and LLM usage, open:

```
http://localhost:8080
```

### Optional: run local observability

Kortny can run with a lightweight local Phoenix trace UI:

```
make compose-up-observability
```

Open Phoenix at `http://localhost:6006`. Phoenix runs as one optional container
behind the `observability` Compose profile, and persists traces to the
`phoenix-data` Docker volume with SQLite by default. Set
`PHOENIX_SQL_DATABASE_URL` to use a separate Postgres database for Phoenix.

Kortny does not bundle a self-hosted Langfuse stack because that requires a
larger observability deployment: Langfuse web/worker, Postgres, ClickHouse,
Redis, and blob storage. To use Langfuse Cloud or a separate Langfuse instance,
set:

```
OTEL_EXPORTER_OTLP_ENDPOINT=https://cloud.langfuse.com/api/public/otel/v1/traces
OTEL_EXPORTER_OTLP_HEADERS=Authorization=Basic <base64-public-secret>,x-langfuse-ingestion-version=4
LANGFUSE_ENABLED=true
LANGFUSE_HOST=https://cloud.langfuse.com
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
```

Generate the Basic Auth value with:

```
printf 'pk-lf-...:sk-lf-...' | base64
```

### Temporal workflow backend

Kortny's default execution layer is the Postgres-backed task queue plus the
`worker` service. Temporal is not part of the normal self-host/dev path because
the current Temporal workflow is still a shadow/skeleton backend, not primary
task execution.

Normal local startup:

```
docker compose up -d --force-recreate
```

To inspect the Temporal experiment explicitly:

```
docker compose --profile temporal up -d --force-recreate
```

Temporal's local UI is then available at `http://localhost:8233`.

`KORTNY_WORKFLOW_BACKEND` defaults to `inline`. HIG-97 records durable-candidate
handoff events for future work, but real Slack execution remains owned by the
main worker until we deliberately migrate the execution layer.

Scheduled work is still owned by Kortny/Postgres in this local stack. The
`scheduler` service materializes due `schedules` rows into normal `tasks` rows;
Temporal Schedules are deferred until Temporal runs with production persistence.

### Sandboxed code execution

Kortny starts an internal `sandbox-runner` service by default. The worker never
mounts the Docker socket directly; it calls the runner over the Compose network,
and the runner talks to Docker through `sandbox-docker-proxy`.

The first sandboxed tool is `code_exec`, a short Python execution tool for
calculations and tiny script checks. It is available to regular employees when
the runner is healthy, but every run requires requester approval in Slack before
execution.

Default sandbox policy:

- Runtime: hardened Docker container launched by `sandbox-runner`.
- Image: `ghcr.io/astral-sh/uv:python3.11-bookworm-slim`.
- Network: disabled with `NetworkMode=none`.
- Filesystem: no host bind mount for user code; the container gets tmpfs
  workspace paths.
- Privileges: no privileged mode, `no-new-privileges`, all capabilities dropped,
  readonly root filesystem.
- Limits: 1 CPU, 512 MB memory, 128 PID cap, 60 second runner default timeout.
- Audit: sandbox lifecycle and bounded stdout/stderr result previews are written
  to the task timeline.

This replaces the earlier "temp dir plus process" direction for untrusted code.
There was no previous production `code_exec` subprocess baseline in Kortny:
fixed tools like PDF generation used trusted in-process library calls. The
current benchmark baseline is therefore operational, not feature parity against
an old code tool: the normal Compose path starts the runner, the worker-to-runner
HTTP bridge executes a small Python smoke check in a sibling sandbox container,
and the runner cleans up containers after execution.

To disable sandbox execution while leaving the service visible for health
checks:

```
KORTNY_SANDBOX_EXECUTION_ENABLED=false
```

### 5. Invite your bot to a channel

Once the Slack ingress service is running:

```
/invite @your-bot-name
```

Say hello:

```
@your-bot-name summarize the last 7 days of this channel
```

Your AI coworker is live.

---

## Features

- **Durable task execution** — every Slack request becomes a tracked
  task with steps, tool calls, and a full audit log.
- **Parallel task processing** — multiple team members can use Kortny
  at once; each task runs independently.
- **File editing and generation** — read, edit, and post files back
  in-thread without leaving Slack.
- **Sandboxed code execution** — employees can ask Kortny to run small
  Python snippets in an isolated no-network container with resource limits;
  requester approval is required before execution.
- **Workspace memory** — structured state and episodic recall across
  conversations and tasks.
- **Per-channel and per-user profiles** — tone, verbosity, approval
  behavior, and proactivity per context.
- **Cost dashboard** — per-task token usage and cost tracking in the
  management UI.
- **100+ integrations** — via Composio OAuth, no manual per-tool
  token setup.
- **Composio-free path** — bring your own MCP servers for a fully
  self-contained integration plane *(V1.1)*.
- **Ambient workspace intelligence** — notices recurring patterns and
  suggests automations unprompted *(V1.1)*.
- **Scheduled tasks** — natural-language scheduling *(V1.1)*.
- **Approval gates** — reaction-based confirmations for sensitive
  actions *(V1.1)*.

---

## Development

Kortny uses `uv` for Python dependency management and local tooling.

```sh
uv sync
```

Common commands:

```sh
make lint          # ruff check
make format        # ruff format
make typecheck     # mypy
make test          # pytest
make check         # lint, format-check, typecheck, and test
make playground    # adk web .
```

Optional local hooks:

```sh
uv run pre-commit install
```

---

## Contributing

Kortny is early and contributions are welcome. Read
[CONTRIBUTING.md](./CONTRIBUTING.md) to get started — whether that's
adding a native tool, improving docs, or reporting a bug.

---

## License

[Apache-2.0](./LICENSE)
