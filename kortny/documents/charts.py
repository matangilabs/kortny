"""Compile a Chart IR block to a themed Vega-Lite spec and render it (HIG-244).

The agent emits a compact, constrained ``Chart`` block (never raw Vega-Lite).
This module compiles it to a valid Vega-Lite spec — applying the document theme
(fonts + colour tokens) and a colour-blind-safe categorical palette — then
renders it with ``vl-convert`` (Vega-Lite via embedded V8; no browser, no Node,
offline). One spec renders to SVG (vector, for Typst PDF) or PNG (for the Office
formats), so a single declarative spec embeds everywhere.

Because we generate the Vega-Lite ourselves, the spec is structurally valid by
construction — the agent never touches Vega-Lite fields, so the only failure
surface is bad input data (caught by the IR schema), not hallucinated chart API.
"""

from __future__ import annotations

from typing import Any

import vl_convert as vlc

from kortny.documents.ir import Chart
from kortny.documents.themes import Theme

# Okabe-Ito colour-blind-safe palette; the brand accent leads so single- and
# few-series charts read on-brand, with CB-safe distinct hues after it.
_OKABE_ITO = (
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
)
_CHART_W = 640
_CHART_H = 380


class ChartRenderError(RuntimeError):
    """vl-convert failed to render the chart spec."""


def _palette(theme: Theme) -> list[str]:
    accent = theme.colors.accent
    return [accent, *[c for c in _OKABE_ITO if c.lower() != accent.lower()]]


def _x_is_quantitative(chart: Chart) -> bool:
    return all(
        isinstance(point.x, (int, float)) and not isinstance(point.x, bool)
        for series in chart.series
        for point in series.points
    )


def compile_chart_spec(chart: Chart, theme: Theme) -> dict[str, Any]:
    """Compile a ``Chart`` block + theme into a themed Vega-Lite spec dict."""

    c = theme.colors
    multi = len(chart.series) > 1
    values: list[dict[str, Any]] = []
    for series in chart.series:
        for point in series.points:
            values.append({"x": point.x, "y": point.y, "series": series.name})

    config: dict[str, Any] = {
        "font": theme.body_font,
        "background": "transparent",
        "view": {"stroke": "transparent"},
        "range": {"category": _palette(theme)},
        "axis": {
            "labelColor": c.muted,
            "titleColor": c.ink,
            "domainColor": c.rule,
            "gridColor": c.rule,
            "tickColor": c.rule,
            "labelFont": theme.body_font,
            "titleFont": theme.body_font,
        },
        "legend": {
            "labelColor": c.ink,
            "titleColor": c.muted,
            "labelFont": theme.body_font,
            "titleFont": theme.mono_font,
        },
        "title": {
            "color": c.ink,
            "font": theme.display_font,
            "fontSize": 16,
            "fontWeight": 700,
            "anchor": "start",
        },
    }

    spec: dict[str, Any] = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "width": _CHART_W,
        "height": _CHART_H,
        "data": {"values": values},
        "config": config,
    }
    if chart.title:
        spec["title"] = chart.title

    if chart.chart_type == "pie":
        spec["mark"] = {"type": "arc", "outerRadius": min(_CHART_W, _CHART_H) // 2 - 20}
        spec["encoding"] = {
            "theta": {"field": "y", "type": "quantitative", "stack": True},
            "color": {
                "field": "x",
                "type": "nominal",
                "legend": {"title": chart.x_label},
            },
        }
        return spec

    mark = {"bar": "bar", "line": "line", "area": "area", "scatter": "point"}[
        chart.chart_type
    ]
    x_type = "quantitative" if _x_is_quantitative(chart) else "nominal"
    x_enc: dict[str, Any] = {"field": "x", "type": x_type, "title": chart.x_label}
    if x_type == "nominal":
        # Preserve the author's category order instead of Vega-Lite's default
        # alphabetical sort — least surprising for hand-authored bar/line data.
        x_enc["sort"] = None
    encoding: dict[str, Any] = {
        "x": x_enc,
        "y": {"field": "y", "type": "quantitative", "title": chart.y_label},
    }
    if multi:
        spec["mark"] = mark
        encoding["color"] = {
            "field": "series",
            "type": "nominal",
            "legend": {"title": None},
        }
    else:
        # Single series: paint it with the brand accent, no legend.
        spec["mark"] = {"type": mark, "color": theme.colors.accent}
    spec["encoding"] = encoding
    return spec


def render_chart_svg(chart: Chart, theme: Theme) -> str:
    """Render the chart to an SVG string (vector — used for the Typst PDF)."""

    spec = compile_chart_spec(chart, theme)
    try:
        return vlc.vegalite_to_svg(spec)
    except Exception as exc:  # vl-convert raises its own error types
        raise ChartRenderError(f"chart render failed: {exc}") from exc


def render_chart_png(chart: Chart, theme: Theme, *, scale: float = 2.0) -> bytes:
    """Render the chart to PNG bytes (raster — used for PPTX/DOCX embedding)."""

    spec = compile_chart_spec(chart, theme)
    try:
        return vlc.vegalite_to_png(spec, scale=scale)
    except Exception as exc:
        raise ChartRenderError(f"chart render failed: {exc}") from exc
