<p align="center">
  <img src="./kortny/dashboard/static/assets/kortny_icon.png" width="110" alt="Kortny logo" />
</p>

<h1 align="center">Kortny</h1>

<h3 align="center">The self-hosted AI coworker that lives in your Slack.</h3>

<p align="center">
  Mention it in a thread; it plans, runs real tasks against your tools, builds and executes code in a sandbox, remembers how your team works, and shows you every step and every cent — all on infrastructure you control.
</p>

<p align="center">
  <a href="./LICENSE"><img src="https://img.shields.io/github/license/boffti/kortny?style=for-the-badge&color=blue" alt="License"></a>
  <a href="https://github.com/boffti/kortny/stargazers"><img src="https://img.shields.io/github/stars/boffti/kortny?style=for-the-badge&color=ffcb47" alt="GitHub stars"></a>
  <a href="https://github.com/boffti/kortny/commits/main"><img src="https://img.shields.io/github/commit-activity/m/boffti/kortny?style=for-the-badge" alt="Commit activity"></a>
  <a href="./CONTRIBUTING.md"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen?style=for-the-badge" alt="PRs welcome"></a>
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.11+"></a>
  <a href="https://github.com/astral-sh/uv"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json&style=flat-square" alt="uv"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json&style=flat-square" alt="Ruff"></a>
  <a href="https://www.docker.com/"><img src="https://img.shields.io/badge/docker_compose-ready-2496ED?style=flat-square&logo=docker&logoColor=white" alt="Docker Compose ready"></a>
  <a href="https://github.com/boffti/kortny/commits/main"><img src="https://img.shields.io/github/last-commit/boffti/kortny?style=flat-square" alt="Last commit"></a>
  <a href="https://github.com/boffti/kortny/issues"><img src="https://img.shields.io/github/issues/boffti/kortny?style=flat-square" alt="Open issues"></a>
</p>

<p align="center">
  <a href="#-quickstart">Quickstart</a> ·
  <a href="#-what-can-it-do">Features</a> ·
  <a href="#-security-model">Security</a> ·
  <a href="#-architecture">Architecture</a> ·
  <a href="#-contributing">Contributing</a>
</p>

<!-- HERO ASSET: replace with a dark/light <picture> screenshot or 20s GIF of a
     Slack thread — user @mentions Kortny, it builds a dashboard, posts a preview link back to the thread. Demo workspace only, no real data. -->

---

> [!IMPORTANT]
> Kortny is early and moving fast. **Star the repo** to follow releases — it genuinely helps the project get found.

## 🤔 Why Kortny

Hosted AI coworkers are black boxes: your Slack history and files flow into someone else's cloud, pricing is opaque, and you can't see what the agent actually did. Generic Slack bots answer questions but don't *do* the work.

Kortny is the alternative you run yourself:

- **It finishes what it starts** — every request becomes a tracked task with a full log of every step, tool call, approval, and decision.
- **It computes instead of guessing** — asked for a dashboard, a report, or an analysis, Kortny writes and executes real code in an isolated sandbox, verifies the output, and delivers a file or a live preview link.
- **No surprise bills** — per-task model, token, and cost accounting in the built-in dashboard. BYO LLM keys (OpenAI, Anthropic, OpenRouter).
- **It gets better the longer it's there** — workspace memory, episodic recall, and a knowledge graph built from how your team actually works.
- **Your bot, your identity** — ship it under your own Slack app name and avatar. Kortny runs the brain; you own the face.
- **Apache-2.0, Docker Compose, your Postgres** — no cloud control plane, no vendor in your data path.

## 🚀 Quickstart

> [!NOTE]
> Prerequisites: Docker + Docker Compose, a Slack workspace you can install apps into, and one LLM API key (OpenAI, Anthropic, or OpenRouter). A [Composio](https://composio.dev) key enables the 100+ integration catalog.

```sh
git clone https://github.com/boffti/kortny
cd kortny
cp .env.example .env   # fill in Slack + LLM keys (see below)
docker compose up -d --force-recreate
```

Then invite your bot and put it to work:

```
/invite @your-bot-name
@your-bot-name summarize the last 7 days of this channel
@your-bot-name build me a dashboard from the CSV I just uploaded
```

The operator dashboard is at `http://localhost:8080` — tasks, traces, costs, memory, schedules, and sandbox runs.

<details>
<summary><b>Step 1 — Create your Slack app (~5 minutes)</b></summary>

1. Go to https://api.slack.com/apps → **Create New App** → **From Manifest**
2. Paste the contents of `manifest.json` from this repo
3. Name the bot whatever you want — this is your bot, your brand
4. Optional: upload an avatar. The Kortny icon ships at `kortny/dashboard/static/assets/kortny_icon.png`
5. Install the app to your workspace
6. Copy the **Bot Token** (`xoxb-...`), **App-Level Token** (`xapp-...` with `connections:write` for Socket Mode), and **Signing Secret**
7. Copy the app's **Client ID** and **Client Secret** for dashboard Sign in with Slack, and add `http://localhost:8080/auth/slack/callback` as an OAuth redirect URL

If you update an existing Slack app from this repo's manifest, re-apply the manifest in Slack and reinstall the app so new scopes and event subscriptions take effect.

</details>

<details>
<summary><b>Step 2 — Configure <code>.env</code></b></summary>

```sh
SLACK_BOT_TOKEN=xoxb-...
SLACK_APP_TOKEN=xapp-...
SLACK_SIGNING_SECRET=...
LLM_PROVIDER=openai          # openai | anthropic | openrouter
LLM_API_KEY=sk-...
LLM_MODEL=gpt-4o
COMPOSIO_API_KEY=...
DASHBOARD_AUTH_MODE=hybrid
DASHBOARD_SLACK_CLIENT_ID=...
DASHBOARD_SLACK_CLIENT_SECRET=...
DASHBOARD_SLACK_REDIRECT_URI=http://localhost:8080/auth/slack/callback
```

That's enough for the default stack. Everything else in `.env.example` is optional with sane local defaults: Postgres credentials, scheduler, Witness, sandbox tuning, observability, Temporal.

Worth knowing:

- `BRAVE_SEARCH_API_KEY` enables the built-in web search tool.
- `ENCRYPTION_KEY` is required before saving dashboard-managed secrets.
- `KORTNY_PUBLIC_BASE_URL` + `KORTNY_PREVIEW_SIGNING_SECRET` enable shareable preview links for sandbox-built dashboards and sites.
- `NETLIFY_AUTH_TOKEN` / `VERCEL_TOKEN` enable one-approval site deploys.
- Change `DASHBOARD_PASSWORD` and `DASHBOARD_SESSION_SECRET` before exposing the dashboard beyond localhost (it binds to `127.0.0.1` by default).

</details>

<details>
<summary><b>What <code>docker compose up</code> starts</b></summary>

The default stack is **7 long-running containers + one-shot `migrate`** — every line is something you can see, name, and reason about (no black box):

| Service | What it does | What breaks without it | Profile |
|---|---|---|---|
| `postgres` | All state: tasks, events, memory, costs, schedules | Everything — the whole system | default |
| `migrate` | Runs Alembic migrations, then exits (one-shot) | Schema drifts; app/worker boot against a stale DB | default |
| `app` | Slack Socket Mode ingress (Bolt) | Slack events never become tasks | default |
| `worker` | Task executor — the LLM + tool loop | Tasks queue but never run | default |
| `ambient` | Supervises three near-idle pollers in one process: **scheduler** (materializes due schedules), **witness** (proactive opportunity scanner), **consolidator** (sleep-time memory/graph consolidation) | Scheduled tasks stop firing, no proactive suggestions, memory stops consolidating into the graph — the request path still works | default |
| `dashboard` | Operator UI at `localhost:8080` — tasks, traces, costs, memory, schedules, sandbox runs | No web console (system keeps running headless) | default |
| `sandbox-runner` | HTTP service that launches throwaway code-exec containers | Code execution / the sandboxed coding workbench is unavailable | default |
| `sandbox-docker-proxy` | Restricted Docker-socket proxy the runner reaches the Docker API through | `sandbox-runner` can't start containers | default |
| `phoenix` | Local OTEL trace UI at `localhost:6006` | No local trace UI (tracing still exports if an endpoint is set) | `observability` |
| `temporal` + `temporal-worker` | Experimental durable workflow backend (UI at `localhost:8233`) | Falls back to the inline worker (the default) | `temporal` |

Sandbox code execution is **on by default** — it's the flagship coworker capability, not a profile add-on. To opt out (e.g. a host that doesn't want code exec), set `KORTNY_SANDBOX_EXECUTION_ENABLED=false` or drop the two `sandbox-*` services via a `compose.override.yaml`.

The three pollers used to be three separate containers; they're now supervised threads inside one `ambient` process with per-loop crash isolation and restart backoff. Each loop still guards its work with a Postgres advisory lock, so the split entrypoints (`python -m kortny.scheduler` / `kortny.witness` / `kortny.consolidator`) remain the scale-out path — run any of them as its own container to peel a loop back out, no double-work.

Witness is on by default and intentionally bounded: it only auto-starts non-interruptive, read-only proactive tasks (one per tick), requires channel membership before posting, and never DMs unless `KORTNY_WITNESS_DELIVER_PRIVATE=true`.

</details>

## ✨ What can it do

| Capability | Status |
|---|---|
| Slack mention → durable tracked task with full audit log | ✅ Shipped |
| **Sandboxed coding workbench** — writes & executes code for dashboards, reports, analysis; verifies by running it | ✅ Shipped |
| Shareable preview links for sandbox-built dashboards/sites | ✅ Shipped |
| One-approval deploys to Netlify / Vercel (tokens never enter the sandbox) | ✅ Shipped |
| Per-task cost, token, and model accounting | ✅ Shipped |
| Workspace memory + episodic recall + knowledge graph | ✅ Shipped |
| Scheduled tasks from natural language ("every Monday at 9...") | ✅ Shipped |
| Proactive suggestions from observed activity (Witness) | ✅ Shipped |
| Approval gates — reaction-based confirmation for risky actions | ✅ Shipped |
| 100+ integrations via Composio OAuth (Gmail, GitHub, HubSpot, ...) | ✅ Shipped |
| PDF / document generation | ✅ Shipped |
| Per-channel personality profiles (tone, verbosity, proactivity) | ✅ Shipped |
| OTEL tracing (Phoenix local / Langfuse Cloud) | ✅ Shipped |
| Google ADK agent runtime (`AGENT_RUNTIME=adk`) | 🟡 Beta |
| Temporal durable workflow backend | 🟡 Experimental |
| Bring-your-own MCP servers (Composio-free integration plane) | ⬜ Planned |
| Network-enabled sandbox profile (pip/npm via egress allowlist) | ⬜ Planned |

## 🧠 How Kortny thinks

Kortny splits into two halves: the **request path** (you ask, it answers —
seconds to minutes) and the **ambient stack** (it watches, learns, and acts
unprompted — hours to weeks). The request path is the visible product; the
ambient stack is what makes it a coworker instead of a chatbot. Four systems
form one pipeline: **observe → remember → consolidate → act**.

### Observe — the senses

Messages in channels Kortny sits in (not addressed to it) land as
observation events, gated by per-channel policy (observation on/off,
proactivity level, retention). No LLM runs at ingest. Periodically a channel
**profile** is assessed: Kortny's working impression of what the room is
about — who's in it, what work recurs, what's struggling — with confidence
and evidence.

### Memory & graph — two stores, different trust

- **Facts** are small, explicit, *user-confirmed* truths ("we use AMA
  citation style"). They follow propose → Slack confirm → active; nothing
  automated may ever invalidate one.
- The **knowledge graph** is the machine-inferred web: entities (people,
  projects, decisions, commitments) and edges, with three properties that
  keep it honest:
  - **Lifecycle** — entities are born `candidate`, earn `active`/`confirmed`
    through corroboration, and decay to `stale`/`archived` when nothing
    reinforces them.
  - **Bi-temporal truth** — contradiction never deletes. The old belief gets
    `invalid_at` stamped and a successor is created, so "what did we believe
    on June 10" has a real answer and stale facts don't get confidently
    repeated.
  - **Provenance** — every entity carries evidence rows pointing at the
    exact tasks/episodes/observations it came from. "Why do you think that?"
    always has an answer.
- **Episodes** are per-task experience summaries — the raw feedstock for
  consolidation, retrieved into context by thread/channel/user proximity.

### Consolidation — sleep

If experience just piled up, memory would be an append-only junk drawer.
The consolidator is Kortny's sleep cycle: it runs when there's enough new
material and the workspace is quiet, and turns experience into knowledge
through committed-independently passes — **promotion** (an LLM arbitrates
each episode against existing knowledge: ADD / UPDATE / INVALIDATE / NOOP —
most things are NOOP; chitchat doesn't become memory), candidate
adjudication, duplicate merge, aging, fact reconciliation, retention
hygiene, per-channel **style cards** (so Kortny sounds like the room), and
embedding backfill. Every run is visible on the dashboard with per-pass
counters and LLM cost.

### Witness — the initiative

The only piece that acts unprompted. It reads fresh channel profiles,
extracts opportunity candidates ("this desk posts a manual trading summary
every morning — I could automate that"), dedupes them into reinforcement,
and gates delivery through a deterministic **receptivity score** built from
your accept/dismiss history — staying silent is a scored, logged decision.
Delivery is budgeted (digest DMs, ≤1 channel post/week). Accepting a
recurring suggestion materializes a real schedule with one confirmation —
a "yes" creates a standing behavior, not a one-time answer.

### The flywheel

```
channel chatter ──► Observe (events → profiles)
                          │
       tasks ──► Episodes │
                    │     │
                    ▼     ▼
              Consolidation: promote / merge / invalidate / age
                    │
                    ▼
         Graph + Facts (bi-temporal, evidenced)
              │                        │
              ▼                        ▼
   Context assembly (better answers)   Witness (better suggestions)
                                           │ accept
                                           ▼
                                   Schedules (standing automations)
                                           │
                                           ▼
                                  more tasks → more episodes → loop
```

Every drive in this loop is visible: the witness scan, memory
consolidation, and catalog sync run as pausable **System schedules** on the
dashboard, every graph entity shows its evidence, and every suggestion
carries its reasons. The whole point of self-hosting an AI coworker is that
none of this is a black box.

## 🛡 Security model

Kortny executes model-written code and acts on your tools, so the guardrails are harness-owned, not model-owned. Full details in [SECURITY.md](./SECURITY.md).

- **Sandboxed execution.** Untrusted code never runs in the worker. It runs in dedicated Docker containers: all capabilities dropped, read-only root filesystem, **no network**, CPU/memory/PID caps, per-task workspaces removed after use. Workers reach Docker only through a restricted socket proxy.
- **Human approval.** Sandbox sessions, external writes, and deploys require explicit requester approval in Slack (react ✅) before execution.
- **Secrets stay out of reach.** Integration and deploy tokens live on the trusted worker. Sandboxed code — and the model — never see them.
- **Execution guardrails.** Max turns, tool-call budgets, and a circuit breaker for repeated failures, enforced by the harness.
- **Everything is auditable.** Every tool call, sandbox command, approval, and output preview lands in the task timeline, inspectable in the dashboard.

### What stays local, what leaves

| Stays on your server | Leaves your server (you choose to whom) |
|---|---|
| Slack event handling (your bot token) | **LLM inference** → your provider (OpenAI / Anthropic / OpenRouter), your keys |
| All tasks, step logs, approvals, audit trail (Postgres) | **External tool actions** → Composio (OAuth broker; credentials are brokered, the model never sees raw tokens) |
| Memory, episodes, knowledge graph | **Web search** → Brave (only if you enable it) |
| Cost & usage accounting | **Deploys** → Netlify/Vercel (only on explicit approved request) |
| Sandbox execution & artifacts | |
| Dashboard & traces | |

There is no Kortny cloud in the data path — no telemetry, no phone-home. If you need a fully self-contained integration plane, the bring-your-own-MCP path is on the roadmap.

## 📦 Production deployment

The Quickstart stack bind-mounts your source and builds nothing — perfect for hacking, wrong for a server. For production, run the published GHCR images with the hardened overlay:

```sh
export KORTNY_VERSION=v1.2.3
docker compose -f compose.yaml -f compose.prod.yaml pull
docker compose -f compose.yaml -f compose.prod.yaml up -d
```

The overlay swaps every service to `ghcr.io/boffti/kortny` (non-root image, no source mount), forces secure cookies, sets restart policies and memory limits, and arms a **startup secret guard** that refuses to boot with placeholder secrets (`ENCRYPTION_KEY`, `DASHBOARD_SESSION_SECRET`, `DASHBOARD_PASSWORD`). Front the localhost-bound dashboard with a reverse proxy.

Full guide — images, required env, reverse proxy (Caddy + nginx), backups, upgrades, and sandbox container GC — in **[docs/deployment.md](./docs/deployment.md)**.

## 🏗 Architecture

```mermaid
flowchart LR
    subgraph YOUR["Your server — docker compose"]
        APP["Slack ingress<br/>(Bolt, Socket Mode)"]
        PG[("Postgres<br/>tasks · memory · costs")]
        WORKER["Worker<br/>LLM + tool loop"]
        AMBIENT["Ambient<br/>scheduler · witness · consolidator"]
        DASH["Dashboard :8080<br/>tasks · traces · costs · previews"]
        RUNNER["Sandbox runner"]
        SBX["Hardened sandbox containers<br/>no network · caps dropped"]
    end
    SLACK(("Slack")) <--> APP
    APP --> PG
    WORKER --> PG
    AMBIENT --> PG
    DASH --> PG
    WORKER --> RUNNER --> SBX
    WORKER -.->|"inference (your keys)"| LLM(("LLM provider"))
    WORKER -.->|"tool actions"| COMPOSIO(("Composio"))
    WORKER -.->|"approved deploys"| HOSTS(("Netlify / Vercel"))
```

How a task flows: a Slack mention (or schedule, or Witness suggestion) becomes a `Task` row → the worker leases it from the Postgres queue → the agent loop plans, calls tools, pauses for approvals, executes code in the sandbox when the work demands computation → the result posts back to the originating thread, and every step is queryable in the dashboard.

<details>
<summary><b>Local development</b></summary>

Kortny uses [`uv`](https://github.com/astral-sh/uv) for dependency management.

```sh
uv sync
make check         # ruff lint + format-check, mypy, pytest
make lint          # ruff check
make typecheck     # mypy
make test          # pytest (unit)
make playground    # adk web .
```

Host-side commands need a local `POSTGRES_URL`:

```sh
export POSTGRES_URL=postgresql://kortny:kortny@localhost:5432/kortny
make migrate
uv run python -m kortny.worker --once     # process one pending task
uv run python -m kortny.witness --once    # one Witness scan/autopilot tick
```

DB-backed integration tests require a **dedicated** test database (the harness refuses to run against the dev DB):

```sh
docker compose exec postgres createdb -U kortny kortny_test
KORTNY_TEST_POSTGRES_URL=postgresql://kortny:kortny@localhost:5432/kortny_test uv run pytest
```

Optional pre-commit hooks: `uv run pre-commit install`.

</details>

<details>
<summary><b>Observability (Phoenix / Langfuse)</b></summary>

Local trace UI with one extra container:

```sh
make compose-up-observability   # Phoenix at http://localhost:6006
```

For Langfuse Cloud (or a separate Langfuse instance), set:

```sh
OTEL_EXPORTER_OTLP_ENDPOINT=https://cloud.langfuse.com/api/public/otel/v1/traces
OTEL_EXPORTER_OTLP_HEADERS=Authorization=Basic <base64 pk:sk>,x-langfuse-ingestion-version=4
LANGFUSE_ENABLED=true
LANGFUSE_HOST=https://cloud.langfuse.com
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
```

</details>

<details>
<summary><b>Sandboxed code execution — policy details</b></summary>

The worker never mounts the Docker socket. It calls the internal `sandbox-runner` service over the Compose network, which reaches the Docker API through `sandbox-docker-proxy` (BUILD/VOLUMES/SYSTEM/SECRETS endpoints disabled).

Two execution modes, both behind requester approval in Slack:

- **`code_exec`** — one-shot Python snippets for calculations and checks.
- **Workbench sessions** — a persistent hardened container per task that the agent drives with `sandbox_bash` and file tools to build apps, run analysis, and produce artifacts. One approval covers the whole session.

Container policy: `ghcr.io/astral-sh/uv:python3.11-bookworm-slim`, `NetworkMode=none`, all capabilities dropped, `no-new-privileges`, read-only root filesystem, CPU/memory/PID limits, per-task workspace volumes removed with the container, idle/TTL reaping, lifecycle + bounded output previews written to the task timeline.

Kill switch: `KORTNY_SANDBOX_EXECUTION_ENABLED=false`.

</details>

## 🗺 Roadmap

Planned, not yet shipped — sequenced toward a stable V1.1:

- **Bring-your-own MCP servers** — fully self-contained integration plane, zero third-party broker.
- **Network-enabled sandbox profile** — `pip`/`npm` installs through an egress proxy with a registry allowlist, unlocking full app scaffolding.
- **Production deployment hardening** — published Docker image, hardened compose variant, CI.
- **Richer document generation** — polished multi-page PDFs and decks.

Follow the [issues](https://github.com/boffti/kortny/issues) for the live backlog.

## 🤝 Contributing

Contributions are welcome — native tools, docs, bug reports, integrations. Start with [CONTRIBUTING.md](./CONTRIBUTING.md). New tools implement one small `Tool` interface plus a catalog metadata entry; the [tool authoring guide](./CLAUDE.md#tool-authoring) walks through it.

Found a security issue? Please use [private vulnerability reporting](./SECURITY.md) — not a public issue.

## ⭐ Star History

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=boffti%2Fkortny&theme=dark&type=Date">
  <img width="100%" src="https://api.star-history.com/svg?repos=boffti%2Fkortny&type=Date" alt="Star History Chart">
</picture>

## 👥 Contributors

<a href="https://github.com/boffti/kortny/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=boffti/kortny" alt="Contributors" />
</a>

## ❓ Why "Kortny"?

Every team has that coworker who quietly keeps everything running — files the report, remembers the decision from three months ago, ships the dashboard nobody else had time for. Kortny is that coworker, except it never sleeps and it runs on your hardware.

## 📄 License

[Apache-2.0](./LICENSE) — permissive, with an explicit patent grant. Build whatever you want with it.
