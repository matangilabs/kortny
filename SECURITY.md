# Security Policy

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security problems.

Report vulnerabilities privately via GitHub's [private vulnerability
reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
on this repository ("Security" tab → "Report a vulnerability").

You can expect an acknowledgement within 72 hours. Please include
reproduction steps and the deployment configuration involved (compose
profile, relevant env vars — redact secrets).

## Supported versions

Kortny is pre-1.0. Only the latest release (and `main`) receives
security fixes.

## Deployment security model

Kortny is self-hosted; the operator owns the trust boundary. Key
properties of the default `docker compose` deployment:

- **Secrets** (Slack tokens, LLM keys, integration tokens) live in
  `.env` on the host and are injected into trusted services only. They
  are never passed into sandbox containers.
- **The dashboard** binds to `127.0.0.1` by default and requires
  authentication. Change `DASHBOARD_PASSWORD` and
  `DASHBOARD_SESSION_SECRET` before exposing it.
- **Postgres** holds all state (tasks, events, memory). Protect it like
  any production database.

## Sandbox threat model (code execution)

Kortny can execute model-written code on behalf of users. That code is
treated as **untrusted by design** — the model can be prompt-injected by
any Slack message, file, or web page it reads. Controls:

**Isolation.** All untrusted code runs in dedicated Docker containers
launched by the `sandbox-runner` service, never in the worker process:

- All Linux capabilities dropped, `no-new-privileges`, read-only root
  filesystem, private IPC, PID/CPU/memory limits (swap disabled).
- **No network** (`NetworkMode=none`). Code cannot reach the internet,
  the host, other containers, or cloud metadata endpoints.
- No host bind mounts. Workspaces are per-task anonymous volumes,
  removed with the container.
- Workers talk to Docker only through `sandbox-runner`, which reaches
  the Docker API via a restricted socket proxy (no BUILD, VOLUMES,
  SYSTEM, or SECRETS endpoints). The raw Docker socket is never mounted
  into worker-facing services.
- Session containers are reaped on idle timeout and hard TTL.

**Human approval.** Every sandbox workbench session and every external
deployment requires explicit requester approval in Slack before
execution.

**Egress of results.** Files leave the sandbox only through audited
export paths (Slack artifacts, signed preview URLs). Preview URLs are
HMAC-signed capability links scoped to one task and directory.
Deployment credentials (Netlify/Vercel tokens) are used on the trusted
worker only — deploys upload sandbox *output*, never grant the sandbox
the token.

**Known limitations (accepted residual risk).**

- Hardened shared-kernel containers are the isolation floor, not a
  hypervisor boundary. A kernel 0-day could escape. Operators who run
  untrusted multi-tenant workloads should consider gVisor (`runsc`) or
  microVM runtimes.
- The audit trail (task events, lifecycle, bounded stdout/stderr
  previews) is the primary detection mechanism; review the dashboard
  trace for unexpected sandbox activity.

## Prompt injection

Kortny treats all Slack content, files, and tool outputs as untrusted
input. Guardrails are harness-owned (approval gates, tool scoping,
execution budgets), not model-owned. Hardening is ongoing work; reports
of practical injection chains that cross an approval or isolation
boundary are very welcome through the private reporting channel above.
