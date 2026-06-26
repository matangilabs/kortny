# Demo workspace seeders

Kortny ships two complementary scripts for setting up a realistic demo workspace. Both live in `scripts/demo/` and use the Northwind B2B SaaS fixture story: five personas (Dana, Priya, Marco, Lena, Theo) posting standups, weekly status updates, ops alerts, a vendor decision thread, a file share, and a v2 launch check-in.

## DB seeder (`scripts/demo/db_seed.py`)

The DB seeder writes backdated observation events, synthetic tasks, episodes, and a channel profile directly into the Postgres database. No Slack messages are posted and no LLM is called. Use this to exercise the ambient stack (observe, witness, automation) without waiting for real team activity to accumulate.

Run it against the live dev database inside compose:

```bash
# Seed 21 days of history into a channel
make seed-sim CHANNEL=C0123456789

# Check current simulator row counts
make status-sim

# Remove all simulator-seeded rows
make clean-sim
```

Or invoke directly:

```bash
docker compose exec worker uv run python -m scripts.demo.db_seed seed --channel C0123456789 --days 21
docker compose exec worker uv run python -m scripts.demo.db_seed status
docker compose exec worker uv run python -m scripts.demo.db_seed clean
```

The `seed` command is idempotent: re-running with the same arguments creates no new observation events or tasks, but bumps the channel profile version so the witness runner treats it as scan-due again.

## Slack seeder (`scripts/demo/slack_seed.py`)

The Slack seeder posts the same fixture story to a real Slack workspace using `chat.postMessage`. It is dry-run by default and will not post unless `--i-understand-this-posts-to-real-slack` is passed explicitly.

WARNING: only run this against a dedicated demo workspace. Never run it against the connected enterprise workspace.

Dry-run (safe, prints what would be posted):

```bash
uv run python -m scripts.demo.slack_seed \
    --token xoxb-... \
    --channels general=C0123,engineering=C0456,product=C0789,ops=C0012,launch=C0345
```

Live post (requires the confirmation flag):

```bash
uv run python -m scripts.demo.slack_seed \
    --token xoxb-... \
    --channels general=C0123,engineering=C0456,product=C0789,ops=C0012,launch=C0345 \
    --i-understand-this-posts-to-real-slack
```

You can also set `SLACK_DEMO_TOKEN` and `SLACK_DEMO_CHANNELS` in the environment instead of passing flags. The script never falls back to `SLACK_BOT_TOKEN`.
