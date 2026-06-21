"""Labeled DocumentSpec cases for the presentation eval (HIG-280).

Each case is a structured DocumentSpec + whether it is expected to have
presentation defects. Covers the high-stakes failure modes: missing chart
labels, wide overflow tables, long stat-card values, and well-formed clean
documents that should score high.

Expand from real agent outputs over time; this seed is the floor the render
pipeline is held to.
"""

from __future__ import annotations

from dataclasses import dataclass

from kortny.documents.ir import (
    Chart,
    ChartPoint,
    ChartSeries,
    DocKind,
    DocumentSpec,
    Heading,
    Prose,
    StatCard,
    StatCards,
    Table,
)


@dataclass(frozen=True, slots=True)
class PresentationCase:
    name: str
    spec: DocumentSpec
    notes: str
    expects_defects: bool = False


SEED_PRESENTATION_CASES: tuple[PresentationCase, ...] = (
    # 1. Clean one-page report: heading + prose + one labeled chart -> no defects
    PresentationCase(
        name="clean_one_page_report",
        spec=DocumentSpec(
            doc_kind=DocKind.report,
            title="Q2 Revenue Summary",
            blocks=[
                Heading(text="Q2 Revenue Summary"),
                Prose(
                    text="This report summarises revenue performance for Q2. Overall results exceeded targets by 12%."
                ),
                Chart(
                    chart_type="bar",
                    title="Monthly Revenue",
                    x_label="Month",
                    y_label="Revenue ($k)",
                    series=[
                        ChartSeries(
                            name="Revenue",
                            points=[
                                ChartPoint(x="Apr", y=120.0),
                                ChartPoint(x="May", y=145.0),
                                ChartPoint(x="Jun", y=160.0),
                            ],
                        )
                    ],
                ),
            ],
        ),
        notes="Well-formed report: heading, prose, labeled bar chart. No defects expected.",
        expects_defects=False,
    ),
    # 2. Wide table that should overflow: many columns -> defects expected
    PresentationCase(
        name="wide_table_overflow",
        spec=DocumentSpec(
            doc_kind=DocKind.report,
            title="Wide Table Report",
            blocks=[
                Heading(text="Detailed Metrics"),
                Prose(
                    text="The following table contains many columns and may overflow the page."
                ),
                Table(
                    caption="Multi-column data",
                    columns=[
                        "Col A",
                        "Col B",
                        "Col C",
                        "Col D",
                        "Col E",
                        "Col F",
                        "Col G",
                        "Col H",
                        "Col I",
                        "Col J",
                        "Col K",
                        "Col L",
                    ],
                    rows=[
                        [
                            "v1",
                            "v2",
                            "v3",
                            "v4",
                            "v5",
                            "v6",
                            "v7",
                            "v8",
                            "v9",
                            "v10",
                            "v11",
                            "v12",
                        ],
                        [
                            "w1",
                            "w2",
                            "w3",
                            "w4",
                            "w5",
                            "w6",
                            "w7",
                            "w8",
                            "w9",
                            "w10",
                            "w11",
                            "w12",
                        ],
                    ],
                ),
            ],
        ),
        notes="Table with 12 columns likely overflows a standard page. Defects expected.",
        expects_defects=True,
    ),
    # 3. Chart with no title or axis labels -> defects expected
    PresentationCase(
        name="chart_no_labels",
        spec=DocumentSpec(
            doc_kind=DocKind.report,
            title="Unlabelled Chart Report",
            blocks=[
                Heading(text="Performance Data"),
                Prose(text="The chart below shows performance trends over the period."),
                Chart(
                    chart_type="line",
                    title=None,
                    x_label=None,
                    y_label=None,
                    series=[
                        ChartSeries(
                            name="series1",
                            points=[
                                ChartPoint(x=1.0, y=10.0),
                                ChartPoint(x=2.0, y=20.0),
                                ChartPoint(x=3.0, y=15.0),
                            ],
                        )
                    ],
                ),
            ],
        ),
        notes="Chart has no title, no x_label, no y_label -- lint catches chart_missing_title and chart_missing_axis_labels.",
        expects_defects=True,
    ),
    # 4. Dense StatCards with long note values -> defects expected
    PresentationCase(
        name="stat_cards_long_values",
        spec=DocumentSpec(
            doc_kind=DocKind.report,
            title="KPI Dashboard",
            blocks=[
                Heading(text="Key Performance Indicators"),
                Prose(
                    text="The metrics below represent the key indicators for this period."
                ),
                StatCards(
                    cards=[
                        StatCard(
                            value="A" * 45,  # exceeds _MAX_CARD_VALUE_CHARS=40
                            label="Revenue",
                            note="This note contains a lot of context that may not fit the tile design properly.",
                        ),
                        StatCard(value="82%", label="NPS", note=None),
                    ]
                ),
            ],
        ),
        notes="First stat card value is too long (45 chars > 40 max), triggering long_stat_value warning.",
        expects_defects=True,
    ),
    # 5. Multi-section report: headings + prose + labeled chart -> no defects
    PresentationCase(
        name="multi_section_report",
        spec=DocumentSpec(
            doc_kind=DocKind.report,
            title="Annual Business Review",
            blocks=[
                Heading(text="Executive Summary"),
                Prose(
                    text="FY2025 delivered record results across all segments. The company achieved 35% YoY growth."
                ),
                Heading(text="Revenue Breakdown"),
                Prose(
                    text="Enterprise accounted for 60% of total revenue while SMB contributed 40%."
                ),
                Chart(
                    chart_type="bar",
                    title="Revenue by Segment",
                    x_label="Segment",
                    y_label="Revenue ($M)",
                    series=[
                        ChartSeries(
                            name="FY2025",
                            points=[
                                ChartPoint(x="Enterprise", y=60.0),
                                ChartPoint(x="SMB", y=40.0),
                            ],
                        )
                    ],
                ),
                Heading(text="Outlook"),
                Prose(
                    text="Management expects 25% growth in FY2026 driven by international expansion."
                ),
            ],
        ),
        notes="Well-structured multi-section report with labeled charts. No defects expected.",
        expects_defects=False,
    ),
    # 6. Pitch deck style: minimal, clean -> no defects
    PresentationCase(
        name="pitch_deck_clean",
        spec=DocumentSpec(
            doc_kind=DocKind.pitch,
            title="Kortny -- AI Coworker for Slack",
            blocks=[
                Heading(text="The Problem"),
                Prose(
                    text="Teams waste hours routing requests, chasing context, and switching tools. Knowledge is siloed and reactive."
                ),
                Heading(text="Our Solution"),
                Prose(
                    text="Kortny is a Slack-native AI coworker that turns conversations into durable tasks with memory and proactivity."
                ),
                StatCards(
                    cards=[
                        StatCard(value="40%", label="Time saved", note=None),
                        StatCard(value="3x", label="Faster resolution", note=None),
                    ]
                ),
            ],
        ),
        notes="Minimal pitch deck with clean stat cards and prose. No defects expected.",
        expects_defects=False,
    ),
)
