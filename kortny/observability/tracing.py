"""OpenTelemetry tracing adapter.

Tracing is intentionally vendor-neutral and no-op safe. Local logs and
task_events remain the primary observability surface unless an OTLP endpoint is
configured.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from opentelemetry import propagate, trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import SpanLimits, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
from opentelemetry.trace import Link, Span, Status, StatusCode

from kortny.config import Settings
from kortny.db.models import Task

logger = logging.getLogger(__name__)

_TRACER_NAME = "kortny"
_DEFAULT_VERSION = "0.1.0"
_MAX_ATTRIBUTE_CHARS = 2_000
_CONFIGURED = False
_TRACING_ENABLED = False
_CAPTURE_CONTENT = "metadata"
_TRACER = trace.get_tracer(_TRACER_NAME, _DEFAULT_VERSION)


def configure_tracing(settings: Settings) -> None:
    """Configure OTEL tracing once for this process.

    If observability is disabled or no OTLP endpoint is configured, this keeps
    the OpenTelemetry API in its default no-op state.
    """

    global _CONFIGURED, _TRACER, _TRACING_ENABLED, _CAPTURE_CONTENT
    if _CONFIGURED:
        return

    _CONFIGURED = True
    # Capture mode gates prompt/response content for BOTH spans and durable
    # task_events rows, so set it before the OTEL-disabled early return.
    _CAPTURE_CONTENT = settings.observability_capture_content
    endpoint = settings.otel_exporter_otlp_endpoint
    if not settings.observability_enabled or endpoint is None:
        _TRACING_ENABLED = False
        return

    version = settings.kortny_version or settings.kortny_release or _DEFAULT_VERSION
    try:
        provider = TracerProvider(
            resource=Resource.create(
                {
                    "service.name": settings.otel_service_name,
                    "service.version": version,
                    "kortny.release": settings.kortny_release or "",
                    "kortny.capture_content": settings.observability_capture_content,
                }
            ),
            sampler=ParentBased(
                root=TraceIdRatioBased(settings.otel_trace_sampling_ratio)
            ),
            span_limits=SpanLimits(
                max_attributes=96,
                max_events=64,
                max_links=16,
                max_span_attribute_length=_MAX_ATTRIBUTE_CHARS,
            ),
        )
        provider.add_span_processor(
            BatchSpanProcessor(
                OTLPSpanExporter(
                    endpoint=endpoint,
                    headers=_parse_otlp_headers(settings.otel_exporter_otlp_headers),
                )
            )
        )
        trace.set_tracer_provider(provider)
        _TRACER = trace.get_tracer(_TRACER_NAME, version)
        _TRACING_ENABLED = True
        logger.info(
            "otel tracing configured service=%s endpoint=%s sampling_ratio=%s",
            settings.otel_service_name,
            endpoint,
            settings.otel_trace_sampling_ratio,
        )
    except Exception:
        _TRACING_ENABLED = False
        logger.exception("otel tracing configuration failed; continuing without traces")


def tracing_enabled() -> bool:
    """Return whether this process is exporting traces."""

    return _TRACING_ENABLED


def capture_content_mode() -> str:
    """Return the active prompt/response capture mode for this process.

    One of ``"metadata"`` (default), ``"summaries"``, or ``"full"``. Set from
    ``OBSERVABILITY_CAPTURE_CONTENT`` at process boot via ``configure_tracing``.
    """

    return _CAPTURE_CONTENT


def start_span(
    name: str,
    *,
    task: Task | None = None,
    attributes: Mapping[str, Any] | None = None,
    linked_traceparent: str | None = None,
) -> AbstractContextManager[Span | None]:
    """Start a span if tracing is configured, otherwise return a no-op context."""

    if not _TRACING_ENABLED:
        return nullcontext(None)

    span_attributes: dict[str, Any] = {}
    if task is not None:
        span_attributes.update(task_span_attributes(task))
    if attributes:
        span_attributes.update(attributes)

    links = _span_links(linked_traceparent)
    return _TRACER.start_as_current_span(
        name,
        attributes=sanitize_span_attributes(span_attributes),
        links=links or None,
    )


def set_span_attributes(attributes: Mapping[str, Any]) -> None:
    """Set attributes on the current span when it is valid."""

    span = trace.get_current_span()
    if not _span_is_recording(span):
        return
    span.set_attributes(sanitize_span_attributes(attributes))


def add_span_event(name: str, attributes: Mapping[str, Any] | None = None) -> None:
    """Add a structured event to the current span."""

    span = trace.get_current_span()
    if not _span_is_recording(span):
        return
    span.add_event(name, attributes=sanitize_span_attributes(attributes or {}))


def record_span_exception(exc: BaseException) -> None:
    """Record an exception on the current span and mark it as errored."""

    span = trace.get_current_span()
    if not _span_is_recording(span):
        return
    span.record_exception(exc)
    span.set_status(Status(StatusCode.ERROR, str(exc)))


def current_trace_context() -> dict[str, str] | None:
    """Return current trace/span IDs for log and task_event correlation."""

    span_context = trace.get_current_span().get_span_context()
    if not span_context.is_valid:
        return None
    return {
        "trace_id": f"{span_context.trace_id:032x}",
        "span_id": f"{span_context.span_id:016x}",
    }


def current_traceparent() -> str | None:
    """Return a W3C traceparent carrier value for async process handoff."""

    if current_trace_context() is None:
        return None
    carrier: dict[str, str] = {}
    propagate.inject(carrier)
    traceparent = carrier.get("traceparent")
    return traceparent or None


def task_span_attributes(task: Task) -> dict[str, Any]:
    """Return stable task correlation attributes for spans."""

    session_id = task.slack_thread_ts or task.slack_message_ts
    return {
        "kortny.task.id": task.id,
        "kortny.installation.id": task.installation_id,
        "slack.channel_id": task.slack_channel_id,
        "slack.thread_ts": task.slack_thread_ts,
        "slack.message_ts": task.slack_message_ts,
        "slack.user_id": task.slack_user_id,
        "langfuse.trace.name": "kortny.task",
        "langfuse.user.id": task.slack_user_id,
        "langfuse.session.id": f"{task.slack_channel_id}:{session_id}"
        if session_id
        else task.slack_channel_id,
        "langfuse.trace.metadata.task_id": task.id,
        "langfuse.trace.metadata.installation_id": task.installation_id,
        "langfuse.trace.metadata.slack_channel_id": task.slack_channel_id,
        "langfuse.trace.metadata.slack_thread_ts": task.slack_thread_ts,
    }


def sanitize_span_attributes(attributes: Mapping[str, Any]) -> dict[str, Any]:
    """Return values accepted by OpenTelemetry span attributes."""

    return {
        str(key): _span_attribute_value(value)
        for key, value in attributes.items()
        if value is not None
    }


def _span_links(traceparent: str | None) -> list[Link]:
    if not traceparent:
        return []

    parent_context = propagate.extract({"traceparent": traceparent})
    span_context = trace.get_current_span(parent_context).get_span_context()
    if not span_context.is_valid:
        return []
    return [Link(span_context)]


def _span_attribute_value(value: Any) -> str | bool | int | float:
    if isinstance(value, str | bool | int | float):
        return _truncate(value)
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Enum):
        return str(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return _json_attribute(asdict(value))
    if isinstance(value, Mapping):
        return _json_attribute(value)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return _json_attribute(list(value))
    return _truncate(str(value))


def _parse_otlp_headers(value: str | None) -> dict[str, str] | None:
    if value is None:
        return None

    headers: dict[str, str] = {}
    for raw_pair in value.split(","):
        pair = raw_pair.strip()
        if not pair:
            continue
        key, separator, raw_header_value = pair.partition("=")
        if not separator:
            logger.warning("ignoring malformed OTLP header entry key=%s", key.strip())
            continue
        key = key.strip()
        header_value = raw_header_value.strip()
        if not key or not header_value:
            continue
        headers[key] = header_value
    return headers or None


def _json_attribute(value: Any) -> str:
    return _truncate_string(
        json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"))
    )


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(child)
            for key, child in value.items()
            if child is not None
        }
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_json_safe(child) for child in value]
    return str(value)


def _truncate(value: str | bool | int | float) -> str | bool | int | float:
    if not isinstance(value, str) or len(value) <= _MAX_ATTRIBUTE_CHARS:
        return value
    return _truncate_string(value)


def _truncate_string(value: str) -> str:
    if len(value) <= _MAX_ATTRIBUTE_CHARS:
        return value
    return value[: _MAX_ATTRIBUTE_CHARS - 3].rstrip() + "..."


def _span_is_recording(span: Span) -> bool:
    return bool(span.get_span_context().is_valid and span.is_recording())
