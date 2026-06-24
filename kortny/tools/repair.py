"""Tool-call repair harness (HIG-291).

Pure module — no DB / session / Slack imports. All logic is deterministic
and side-effect-free so it can be unit-tested without infrastructure.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from kortny.tools.types import JsonObject, JsonSchema, RecoverableToolError, ToolResult


@dataclass(frozen=True, slots=True)
class ToolRepair:
    phase: Literal["pre_call", "post_call"]
    pattern: str
    arguments: JsonObject
    changed_keys: tuple[str, ...]
    note: str
    retry: bool = False  # True only for post-call reactive retries


@dataclass(frozen=True, slots=True)
class RepairContext:
    tool_name: str
    args: JsonObject
    parameters: JsonSchema | None  # None for Composio (no local schema)
    result: ToolResult | None = None
    error: RecoverableToolError | None = None


Rule = Callable[[RepairContext], ToolRepair | None]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MARKDOWN_LINK_RE = re.compile(r"^\[([^\]]*)\]\(([^)]+)\)$")
_PARSEINT_ERROR_RE = re.compile(
    r"ParseInt|invalid syntax|parsing \"[\d.]+\"", re.IGNORECASE
)


def _schema_properties(parameters: JsonSchema | None) -> JsonObject:
    if parameters is None:
        return {}
    props = parameters.get("properties", {})
    return props if isinstance(props, dict) else {}


def _required_fields(parameters: JsonSchema | None) -> set[str]:
    if parameters is None:
        return set()
    req = parameters.get("required", [])
    return set(req) if isinstance(req, list) else set()


def _normalize_name(s: str) -> str:
    return s.lower().replace("_", "").replace("-", "")


# ---------------------------------------------------------------------------
# PRE-call rules
# ---------------------------------------------------------------------------

_SCHEMA_TYPE_TO_PYTHON: dict[str, type] = {
    "object": dict,
    "array": list,
    "boolean": bool,
    "integer": int,
}


def _rule_parse_json_string_fields(ctx: RepairContext) -> ToolRepair | None:
    """Rule 1 — schema says object/array/boolean/integer/number but arg is a str."""
    if ctx.parameters is None:
        return None

    props = _schema_properties(ctx.parameters)
    repaired = dict(ctx.args)
    changed: list[str] = []

    for key, val in ctx.args.items():
        if not isinstance(val, str):
            continue
        prop_schema = props.get(key)
        if not isinstance(prop_schema, dict):
            continue
        schema_type = prop_schema.get("type")
        if schema_type not in ("object", "array", "boolean", "integer", "number"):
            continue
        try:
            decoded = json.loads(val)
        except (json.JSONDecodeError, ValueError):
            continue

        # Verify the decoded value matches the expected Python type.
        if schema_type == "object" and not isinstance(decoded, dict):
            continue
        if schema_type == "array" and not isinstance(decoded, list):
            continue
        if schema_type == "boolean" and not isinstance(decoded, bool):
            continue
        if schema_type == "integer" and (
            isinstance(decoded, bool) or not isinstance(decoded, int)
        ):
            continue
        if schema_type == "number" and not isinstance(decoded, (int, float)):
            continue

        repaired[key] = decoded
        changed.append(key)

    if not changed:
        return None

    changed_keys = tuple(sorted(changed))
    if len(changed_keys) == 1:
        key = changed_keys[0]
        schema_type = _schema_properties(ctx.parameters).get(key, {}).get("type", "?")
        note = (
            f"Field '{key}' must be {schema_type}, not a JSON string. "
            f"Decoded it for you; send the native type next time."
        )
    else:
        keys_str = ", ".join(f"'{k}'" for k in changed_keys)
        note = (
            f"Fields {keys_str} must be native types, not JSON strings. "
            f"Decoded them for you; send native types next time."
        )

    return ToolRepair(
        phase="pre_call",
        pattern="parse_json_string_fields",
        arguments=repaired,
        changed_keys=changed_keys,
        note=note,
    )


def _rule_omit_null_optional(ctx: RepairContext) -> ToolRepair | None:
    """Rule 2 — drop args whose value is None (skip required fields when schema known)."""
    required = _required_fields(ctx.parameters)
    repaired = {
        k: v for k, v in ctx.args.items() if not (v is None and k not in required)
    }
    changed = [k for k in ctx.args if k not in repaired]
    if not changed:
        return None

    changed_keys = tuple(sorted(changed))
    keys_str = ", ".join(f"'{k}'" for k in changed_keys)
    note = (
        f"Dropped null fields {keys_str}. "
        f"Omit optional fields rather than passing null."
    )
    return ToolRepair(
        phase="pre_call",
        pattern="omit_null_optional",
        arguments=repaired,
        changed_keys=changed_keys,
        note=note,
    )


def _rule_drop_empty_optional_placeholders(ctx: RepairContext) -> ToolRepair | None:
    """Rule 3 — drop args whose value is '', {}, or [] (skip required when schema known)."""
    required = _required_fields(ctx.parameters)
    repaired = {
        k: v
        for k, v in ctx.args.items()
        if not (v in ("", {}, []) and k not in required)
    }
    changed = [k for k in ctx.args if k not in repaired]
    if not changed:
        return None

    changed_keys = tuple(sorted(changed))
    keys_str = ", ".join(f"'{k}'" for k in changed_keys)
    note = (
        f"Dropped empty placeholder fields {keys_str}. "
        f"Omit optional fields instead of passing empty values."
    )
    return ToolRepair(
        phase="pre_call",
        pattern="drop_empty_optional_placeholders",
        arguments=repaired,
        changed_keys=changed_keys,
        note=note,
    )


def _strip_markdown_wrapper(value: str) -> str | None:
    """Return the cleaned string if a markdown wrapper was removed, else None."""
    original = value

    # Strip triple backticks first (``` ... ```)
    if value.startswith("```") and value.endswith("```") and len(value) >= 6:
        inner = value[3:-3].strip()
        # Remove an optional language tag on the first line.
        first_newline = inner.find("\n")
        if first_newline != -1:
            first_line = inner[:first_newline]
            if first_line and " " not in first_line:
                inner = inner[first_newline + 1 :].strip()
        value = inner

    # Strip single backticks (` ... `)
    if (
        value.startswith("`")
        and value.endswith("`")
        and len(value) >= 2
        and not value.startswith("``")
    ):
        value = value[1:-1]

    # Markdown link [text](url) → url
    m = _MARKDOWN_LINK_RE.match(value)
    if m:
        value = m.group(2)

    # Strip surrounding single or double quotes
    if len(value) >= 2 and (
        (value.startswith('"') and value.endswith('"'))
        or (value.startswith("'") and value.endswith("'"))
    ):
        value = value[1:-1]

    return value if value != original else None


def _rule_strip_markdown_string_args(ctx: RepairContext) -> ToolRepair | None:
    """Rule 4 — strip markdown wrappers from string-typed args."""
    props = _schema_properties(ctx.parameters)
    repaired = dict(ctx.args)
    changed: list[str] = []

    for key, val in ctx.args.items():
        if not isinstance(val, str):
            continue
        # Only apply if schema says string OR schema is absent for this key.
        prop_schema = props.get(key) if props else None
        if prop_schema is not None and prop_schema.get("type") != "string":
            continue
        cleaned = _strip_markdown_wrapper(val)
        if cleaned is not None:
            repaired[key] = cleaned
            changed.append(key)

    if not changed:
        return None

    changed_keys = tuple(sorted(changed))
    if len(changed_keys) == 1:
        note = (
            f"Stripped markdown wrapper from field '{changed_keys[0]}'. "
            f"Send raw strings, not markdown-formatted values."
        )
    else:
        keys_str = ", ".join(f"'{k}'" for k in changed_keys)
        note = (
            f"Stripped markdown wrappers from fields {keys_str}. "
            f"Send raw strings, not markdown-formatted values."
        )
    return ToolRepair(
        phase="pre_call",
        pattern="strip_markdown_string_args",
        arguments=repaired,
        changed_keys=changed_keys,
        note=note,
    )


def _rule_coerce_schema_scalars(ctx: RepairContext) -> ToolRepair | None:
    """Rule 5 — coerce string/float args to the schema-declared scalar type."""
    if ctx.parameters is None:
        return None

    props = _schema_properties(ctx.parameters)
    repaired = dict(ctx.args)
    changed: list[str] = []

    for key, val in ctx.args.items():
        prop_schema = props.get(key)
        if not isinstance(prop_schema, dict):
            continue
        schema_type = prop_schema.get("type")

        if schema_type == "integer":
            coerced: object = None
            coerced_set = False
            if isinstance(val, float) and not isinstance(val, bool):
                # 45.0 → 45 (only if integral)
                if val == int(val):
                    coerced = int(val)
                    coerced_set = True
            elif isinstance(val, str):
                stripped = val.strip()
                try:
                    as_float = float(stripped)
                    if as_float == int(as_float):
                        coerced = int(as_float)
                        coerced_set = True
                except ValueError:
                    pass
            if coerced_set:
                repaired[key] = coerced
                changed.append(key)

        elif schema_type == "number":
            if isinstance(val, str):
                stripped = val.strip()
                try:
                    repaired[key] = float(stripped)
                    changed.append(key)
                except ValueError:
                    pass

        elif schema_type == "boolean":
            coerced_bool: bool | None = None
            if isinstance(val, str):
                if val.lower() == "true":
                    coerced_bool = True
                elif val.lower() == "false":
                    coerced_bool = False
            elif isinstance(val, int) and not isinstance(val, bool):
                if val == 1:
                    coerced_bool = True
                elif val == 0:
                    coerced_bool = False
            if coerced_bool is not None:
                repaired[key] = coerced_bool
                changed.append(key)

    if not changed:
        return None

    changed_keys = tuple(sorted(changed))
    # Build a note; cap at ~120 chars.
    note_parts: list[str] = []
    for k in changed_keys:
        prop_schema = props.get(k, {})
        schema_type = prop_schema.get("type", "?")
        from_type = type(ctx.args[k]).__name__
        to_type = type(repaired[k]).__name__
        note_parts.append(
            f"Coerced field '{k}' from {from_type} to {to_type}. "
            f"Schema requires {schema_type}."
        )
    note = " ".join(note_parts)
    if len(note) > 120:
        note = (
            f"Coerced {len(changed_keys)} fields to schema scalar types. "
            f"Schema requires proper scalar types."
        )

    return ToolRepair(
        phase="pre_call",
        pattern="coerce_schema_scalars",
        arguments=repaired,
        changed_keys=changed_keys,
        note=note,
    )


def _rule_alias_native_arg_names(ctx: RepairContext) -> ToolRepair | None:
    """Rule 6 — rename a single unrecognised arg key to a matching schema property."""
    if ctx.parameters is None:
        return None

    props = _schema_properties(ctx.parameters)
    schema_keys = set(props.keys())
    arg_keys = set(ctx.args.keys())

    unknown_arg_keys = arg_keys - schema_keys
    missing_schema_keys = schema_keys - arg_keys

    if len(unknown_arg_keys) != 1 or not missing_schema_keys:
        return None

    unknown_key = next(iter(unknown_arg_keys))
    normalized_unknown = _normalize_name(unknown_key)

    # Find schema properties that match the normalized form.
    candidates = [
        k for k in missing_schema_keys if _normalize_name(k) == normalized_unknown
    ]
    if len(candidates) != 1:
        return None  # ambiguous

    new_key = candidates[0]
    repaired = {(new_key if k == unknown_key else k): v for k, v in ctx.args.items()}
    changed_keys = (unknown_key,)
    note = (
        f"Renamed arg '{unknown_key}' to '{new_key}'. "
        f"Use the exact parameter name from the tool schema."
    )
    return ToolRepair(
        phase="pre_call",
        pattern="alias_native_arg_names",
        arguments=repaired,
        changed_keys=changed_keys,
        note=note,
    )


# ---------------------------------------------------------------------------
# POST-call rules
# ---------------------------------------------------------------------------


def _rule_retry_parseint_float_as_string(ctx: RepairContext) -> ToolRepair | None:
    """Rule 7 — retry when API rejects an integral float (e.g. 45.0) as non-integer."""
    if ctx.error is None:
        return None

    error_text = ctx.error.message or str(ctx.error)
    if not _PARSEINT_ERROR_RE.search(error_text):
        return None

    # Find float args with zero fractional part.
    pairs: list[str] = []
    repaired = dict(ctx.args)
    for key, val in ctx.args.items():
        if isinstance(val, float) and not isinstance(val, bool) and val == int(val):
            int_val = int(val)
            repaired[key] = str(int_val)
            pairs.append(f'{key}: {val!r} -> "{int_val}"')

    if not pairs:
        return None

    changed_keys = tuple(sorted(k for k in repaired if repaired[k] != ctx.args.get(k)))
    pairs_str = ", ".join(pairs)
    note = (
        f"Replaced integral-float values ({pairs_str}) with string form. "
        f"The API requires integer strings, not floats."
    )
    return ToolRepair(
        phase="post_call",
        pattern="retry_parseint_float_as_string",
        arguments=repaired,
        changed_keys=changed_keys,
        note=note,
        retry=True,
    )


# ---------------------------------------------------------------------------
# Rule registries
# ---------------------------------------------------------------------------

PRE_RULES: list[Rule] = [
    _rule_parse_json_string_fields,
    _rule_omit_null_optional,
    _rule_drop_empty_optional_placeholders,
    _rule_strip_markdown_string_args,
    _rule_coerce_schema_scalars,
    _rule_alias_native_arg_names,
]

POST_RULES: list[Rule] = [
    _rule_retry_parseint_float_as_string,
]

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def repair_pre_call(
    *, tool_name: str, args: JsonObject, parameters: JsonSchema | None
) -> ToolRepair | None:
    """Run PRE_RULES in order; return first match."""
    ctx = RepairContext(tool_name=tool_name, args=args, parameters=parameters)
    for rule in PRE_RULES:
        repair = rule(ctx)
        if repair is not None:
            return repair
    return None


def repair_post_call(
    *,
    tool_name: str,
    args: JsonObject,
    result: ToolResult | None,
    error: RecoverableToolError | None,
) -> ToolRepair | None:
    """Run POST_RULES in order; return first match."""
    ctx = RepairContext(
        tool_name=tool_name,
        args=args,
        parameters=None,
        result=result,
        error=error,
    )
    for rule in POST_RULES:
        repair = rule(ctx)
        if repair is not None:
            return repair
    return None
