# Kortny production application image (HIG-200).
#
# Single multi-stage image used by every long-running Python service (slack app,
# worker, ambient, dashboard, sandbox-runner control plane, temporal worker).
# The service is selected by overriding the container command, e.g.
#   command: ["python", "-m", "kortny.worker"]
# The default CMD is the worker.
#
# This is the APP image. The sandbox EXECUTION image (the one throwaway code
# containers boot from) is built separately from docker/sandbox-exec.Dockerfile
# and published as ghcr.io/boffti/kortny-sandbox-exec.

# ---------------------------------------------------------------------------
# Stage 1 — builder: resolve and install deps into a self-contained venv.
# ---------------------------------------------------------------------------
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /app

# Install dependencies first (cached) from the lockfile only, without the
# project source, so dependency layers are reused across source-only changes.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# Now copy the project source and install the project itself into the venv.
COPY kortny/ ./kortny/
COPY alembic.ini ./
COPY README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

# ---------------------------------------------------------------------------
# Stage 2 — runtime: slim image, non-root, venv + source only.
# ---------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS runtime

# OCI image labels (populated by --build-arg in CI).
ARG KORTNY_VERSION=dev
ARG KORTNY_REVISION=unknown
LABEL org.opencontainers.image.source="https://github.com/boffti/kortny" \
      org.opencontainers.image.title="kortny" \
      org.opencontainers.image.description="Self-hosted Slack-native AI coworker" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.version="${KORTNY_VERSION}" \
      org.opencontainers.image.revision="${KORTNY_REVISION}"

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    KORTNY_VERSION="${KORTNY_VERSION}"

WORKDIR /app

# Non-root runtime user with a fixed uid for predictable volume ownership.
RUN groupadd --gid 10001 kortny \
    && useradd --uid 10001 --gid 10001 --no-create-home --home-dir /app kortny

# Copy the resolved venv and the application source from the builder. alembic
# migrations live under kortny/db/migrations and travel with the kortny package.
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app/kortny /app/kortny
COPY --from=builder /app/alembic.ini /app/alembic.ini

# Runtime assets read from the repo root at runtime: the Slack app manifest used
# by the dashboard setup wizard (HIG-209) and migrations config.
COPY manifest.json /app/manifest.json

# Fail-fast secret guard entrypoint.
COPY docker/entrypoint.sh /usr/local/bin/kortny-entrypoint
RUN chmod +x /usr/local/bin/kortny-entrypoint

# Data dirs the worker/dashboard write to (artifacts, embeddings cache); chown
# so the non-root user owns them when no named volume is mounted.
RUN mkdir -p /data/artifacts /data/fastembed-cache \
    && chown -R 10001:10001 /app /data

USER 10001:10001

ENTRYPOINT ["/usr/local/bin/kortny-entrypoint"]
CMD ["python", "-m", "kortny.worker"]
