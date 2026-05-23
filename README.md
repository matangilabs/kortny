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
- A Composio API key

### 1. Create your Slack app

1. Go to https://api.slack.com/apps → Create New App → From Manifest
2. Paste the contents of `slack/manifest.json` from this repo
3. Name your bot whatever you want — this is your bot, your brand
4. Upload a custom avatar if you'd like
5. Install the app to your workspace
6. Copy your **Bot Token** (`xoxb-...`) and **Signing Secret** from
   Basic Information

### 2. Clone and configure

```
git clone https://github.com/boffti/kortny
cd kortny
cp .env.example .env
```

Edit `.env`:

```x
SLACK_BOT_TOKEN=xoxb-...
SLACK_SIGNING_SECRET=...
LLM_PROVIDER=openai          # openai | anthropic | openrouter
LLM_API_KEY=sk-...
LLM_MODEL=gpt-4o             # or claude-3-5-sonnet, etc.
COMPOSIO_API_KEY=...
POSTGRES_URL=postgresql://kortny:kortny@postgres:5432/kortny
```

### 3. Start Kortny

```
docker compose up -d
```

### 4. Complete setup via the management UI

Visit http://localhost:3000:
- Validate your Slack connection
- Test your LLM provider
- Connect your first integration (Gmail, HubSpot, GitHub, etc.)

### 5. Invite your bot to a channel

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

## Documentation

Full documentation lives in [/docs](./docs):

- [Configuration reference](./docs/configuration.md)
- [Adding integrations](./docs/integrations.md)
- [Architecture overview](./docs/architecture.md)
- [Self-hosting guide](./docs/self-hosting.md)

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

MIT
