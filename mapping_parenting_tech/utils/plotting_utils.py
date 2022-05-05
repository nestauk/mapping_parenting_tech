"""
mapping_parenting_tech.utils.plotting_utils
Functions for generating graphs
"""
import altair as alt
import pandas as pd
from typing import Iterator

ChartType = alt.vegalite.v4.api.Chart

# Brand aligned fonts and colours
FONT = "Averta"
TITLE_FONT = "Zosia"
NESTA_COLOURS = [
    # Nesta brand colors:
    "#0000FF",
    "#FDB633",
    "#18A48C",
    "#9A1BBE",
    "#EB003B",
    "#FF6E47",
    "#646363",
    "#0F294A",
    "#97D9E3",
    "#A59BEE",
    "#F6A4B7",
    "#D2C9C0",
    "#000000",
    # Extra non-Nesta colors:
    "#4d16c4",
]


def nestafont():
    """Define Nesta fonts"""
    return {
        "config": {
            "title": {"font": TITLE_FONT, "anchor": "start"},
            "subtitle": {"font": FONT},
            "axis": {"labelFont": FONT, "titleFont": FONT},
            "header": {"labelFont": FONT, "titleFont": FONT},
            "legend": {"labelFont": FONT, "titleFont": FONT},
            "range": {
                "category": NESTA_COLOURS,
                "ordinal": {
                    "scheme": NESTA_COLOURS
                },  # this will interpolate the colors
            },
        }
    }


alt.themes.register("nestafont", nestafont)
alt.themes.enable("nestafont")


def test_chart():
    """Generates a simple test chart"""
    return (
        alt.Chart(
            pd.DataFrame(
                {
                    "labels": ["A", "B", "C"],
                    "values": [10, 15, 30],
                    "label": ["This is A", "This is B", "And this is C"],
                }
            ),
            width=400,
            height=200,
        )
        .mark_bar()
        .encode(
            alt.Y("labels:O", title="Vertical axis"),
            alt.X("values:Q", title="Horizontal axis"),
            tooltip=["label", "values"],
            color="labels",
        )
        .properties(
            title={
                "anchor": "start",
                "text": ["Chart title"],
                "subtitle": ["Longer descriptive subtitle"],
                "subtitleFont": FONT,
            },
        )
        .configure_axis(
            gridDash=[1, 7],
            gridColor="grey",
        )
        .configure_view(strokeWidth=0)
        .interactive()
    )


def bar_chart(
    data: pd.DataFrame,
    values_column: str,
    labels_column: str,
    values_title: str = "Values",
    labels_title: str = "Labels",
    tooltip: Iterator[str] = None,
    color: str = None,
    chart_title: str = "",
    chart_subtitle: str = "",
):
    """Generates a simple bar chart"""
    tooltip = [values_column, labels_column] if tooltip is None else tooltip
    color_ = NESTA_COLOURS[0] if color is None else None
    color = [labels_column] if color is None else color
    return (
        alt.Chart(
            data,
            width=width,
            height=height,
        )
        .mark_bar(color=color_)
        .encode(
            x=alt.X(f"{values_column}:Q", title=labels_title),
            y=alt.Y(f"{labels_column}:N", title=values_title, sort="-x"),
            tooltip=tooltip,
            # color=color,
        )
        .properties(
            title={
                "anchor": "start",
                "text": chart_title,
                "subtitle": chart_subtitle,
                "subtitleFont": FONT,
            },
        )
        .configure_axis(
            gridDash=[1, 7],
            gridColor="grey",
        )
        .configure_view(strokeWidth=0)
        .interactive()
    )
