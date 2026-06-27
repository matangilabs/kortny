# Cross-app Orchestration Eval

Measures whether the agent routes multi-step requests to the correct
integrations, avoids context-leak (answering from stale cached data instead of
calling the API), and stays in scope (does not call irrelevant toolkits).

## Running the eval

### Offline smoke eval — no secrets required

```bash
make eval-smoke
```

Loads the committed `fixtures/smoke_goldens.json`, scores the `smoke=True`
subset of cases against the recorded goldens, and prints a pass/fail report.
Exits non-zero if any smoke case fails.

Needs: nothing. No API keys, no database, no live agent, no Composio
connections. Safe to run in CI.

### Live eval — records new goldens

```bash
make eval
```

Runs every seed case through the real agent against a live install, prints the
report, and writes results back to `fixtures/smoke_goldens.json`. Commit the
updated file so `make eval-smoke` stays in sync.

Needs: a running Postgres instance with a Kortny install, a valid LLM API key
(`LLM_API_KEY` / `LLM_MODEL`), and Composio integrations connected for the
scope user. Optional env overrides:

| Variable | Purpose |
|---|---|
| `KORTNY_EVAL_SCOPE_USER_ID` | Slack user whose connections are used (auto-detected otherwise) |
| `KORTNY_EVAL_SCOPE_CHANNEL_ID` | Channel for channel-scoped connections (defaults to a synthetic sentinel) |

## Adding a case

1. Open `cases.py` and append an `OrchestrationCase` to `SEED_ORCHESTRATION_CASES`.
   - Set `expected_apps` to the Composio toolkit slugs the agent **must** call.
   - Set `must_use_tools=True` when the answer must come from a live API call (not cached context).
   - Set `forbidden_apps` for scope-isolation guards.
   - Set `requires_toolkits` for toolkits not yet connected in the eval workspace; these cases skip cleanly at runtime.
   - Set `smoke=True` if you want the case in the offline smoke subset (you must add a fixture too — see below).

2. Run `make eval` to execute the new case against a live install. The runner
   appends the result to `fixtures/smoke_goldens.json`.

3. Review the recorded fixture. If the agent produced the **correct** behavior,
   commit it. If not, fix the agent and re-run.

4. If you added `smoke=True`, verify `make eval-smoke` passes before pushing.

### Adding a fixture manually

If you want to hand-author a golden (e.g. for a case you cannot run live yet),
add an entry to `fixtures/smoke_goldens.json` directly:

```json
{
  "your case request string": {
    "called_apps": ["github", "linear"],
    "any_tool_called": true,
    "answer": "",
    "skipped": false,
    "skip_reason": null
  }
}
```

The key is the exact `case.request` string. `called_apps` encodes the desired
behavior (the apps the agent **should** call for a correct answer).

## Fixture format

`fixtures/smoke_goldens.json` — a JSON object keyed by `case.request`:

```json
{
  "<request string>": {
    "called_apps": ["<toolkit_slug>", ...],
    "any_tool_called": true | false,
    "answer": "",
    "skipped": false,
    "skip_reason": null
  }
}
```

`called_apps` is a sorted list; the scorer reconstructs a `frozenset` internally.
`answer` is intentionally blank in committed fixtures — the scorer does not
currently assert on answer content.

## Smoke subset

Cases marked `smoke=True` in `cases.py` form the offline regression gate:

- Cross-app explicit (GitHub + Linear)
- Cross-app implicit with context-leak guard
- Cross-app write (Gmail + Google Calendar)
- Single-app status read (Linear)
- Single-app implicit read (Gmail)
- Single-app calendar read (Google Calendar)
- Single-app GitHub PR check
- Cross-app doc-to-ticket (Notion + Linear)
- Finance ticker with scope isolation
- Internal-state guard (schedules, no connected app)
- Pure-knowledge no-tool guard

A case without a committed fixture will be `skipped` by the replay runner and
will **not** fail `make eval-smoke`. The drift-guard test
`test_smoke_cases_have_committed_fixture` in `tests/test_orchestration_eval.py`
fails at pytest time instead, so the gap is caught in CI.
