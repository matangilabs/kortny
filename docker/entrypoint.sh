#!/bin/sh
# Kortny container entrypoint (HIG-200).
#
# Fail-fast secret guard for production. When KORTNY_REQUIRE_SECURE_ENV is
# truthy (the prod compose overlay sets it), this refuses to start any service
# until the SECURITY-CRITICAL configuration is present and not left at the
# shipped placeholder defaults. It deliberately does NOT require Slack / LLM /
# Composio credentials: the dashboard's first-run setup wizard (HIG-209) boots
# precisely when those are absent, walks the operator through provider setup, and
# renders a copyable .env. Hard-failing on them would break that flow.
#
# Outside prod (KORTNY_REQUIRE_SECURE_ENV unset/false) this is a transparent
# pass-through, so the dev compose stack is unaffected.
#
# Var names and defaults verified against:
#   kortny/config/settings.py            -> ENCRYPTION_KEY
#   kortny/dashboard/settings.py         -> DASHBOARD_SESSION_SECRET,
#                                           DASHBOARD_PASSWORD

set -eu

is_truthy() {
    case "$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')" in
        1 | true | yes | on) return 0 ;;
        *) return 1 ;;
    esac
}

if is_truthy "${KORTNY_REQUIRE_SECURE_ENV:-}"; then
    errors=""

    # ENCRYPTION_KEY: secrets at rest (MCP server secrets, provider API keys)
    # are Fernet-encrypted with a key derived from this. Empty = unrecoverable.
    if [ -z "${ENCRYPTION_KEY:-}" ]; then
        errors="${errors}
  - ENCRYPTION_KEY is empty. Set a long random string; secrets at rest are
    encrypted with a key derived from it and cannot be recovered if it changes."
    fi

    # DASHBOARD_SESSION_SECRET: signs dashboard session cookies. The shipped
    # default must never run in prod, and it must be at least 16 chars.
    if [ -z "${DASHBOARD_SESSION_SECRET:-}" ]; then
        errors="${errors}
  - DASHBOARD_SESSION_SECRET is empty. Set a long random string (>= 16 chars)."
    elif [ "${DASHBOARD_SESSION_SECRET}" = "change-me-dashboard-session-secret" ]; then
        errors="${errors}
  - DASHBOARD_SESSION_SECRET is still the default placeholder. Set a unique
    long random string (>= 16 chars)."
    elif [ "${#DASHBOARD_SESSION_SECRET}" -lt 16 ]; then
        errors="${errors}
  - DASHBOARD_SESSION_SECRET is too short. Use at least 16 characters."
    fi

    # DASHBOARD_PASSWORD: bootstrap admin login. The shipped default is public.
    if [ -z "${DASHBOARD_PASSWORD:-}" ]; then
        errors="${errors}
  - DASHBOARD_PASSWORD is empty. Set a strong bootstrap admin password."
    elif [ "${DASHBOARD_PASSWORD}" = "change-me" ]; then
        errors="${errors}
  - DASHBOARD_PASSWORD is still the default 'change-me'. Set a strong one."
    fi

    if [ -n "${errors}" ]; then
        printf '%s\n' "============================================================" >&2
        printf '%s\n' "Kortny refused to start: insecure production configuration." >&2
        printf '%s\n' "============================================================" >&2
        printf '%s\n' "${errors}" >&2
        printf '\n%s\n' "Fix these in your environment / .env, then restart. (Slack, LLM," >&2
        printf '%s\n' "and Composio keys are NOT required here — the dashboard setup" >&2
        printf '%s\n' "wizard handles those on first run.)" >&2
        exit 78  # EX_CONFIG
    fi
fi

exec "$@"
