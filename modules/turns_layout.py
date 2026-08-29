"""Turns tab layout: per-request length histograms with model/role filters.

All Turns dcc.Stores live here.
"""
from __future__ import annotations

from dash import dcc, html

from modules import controls
from modules.theme import F_SMALL, GRAPH_CONFIG
from modules.records import DIMENSIONS


def layout() -> html.Div:
    side = controls.sidebar([
        controls.label("Dataset",
                       info_text="Pick which cached dataset to histogram. "
                                 "Datasets are imported on the Overview tab. "
                                 "An Explorer selection scopes these charts "
                                 "to the selected conversations."),
        controls.dropdown("at-turns-dataset-dd", "pick a cached dataset"),
        controls.label("Models (empty = all)",
                       info_text="Restrict the histograms to requests served "
                                 "by specific models. Empty = every model in "
                                 "the dataset."),
        controls.dropdown("at-turns-models-dd", "all models", multi=True),
        controls.label("Role",
                       info_text="main agent = the top-level conversation's "
                                 "own requests; subagents = nested agents "
                                 "spawned by tool calls, each with its own "
                                 "context."),
        dcc.Checklist(
            id="at-turns-roles-cl",
            options=[{"label": " main agent", "value": "main"},
                     {"label": " subagents", "value": "subagent"}],
            value=["main", "subagent"],
            style={"fontSize": F_SMALL},
        ),
        controls.label("X scale",
                       info_text="log bins: geometric bin edges — right for "
                                 "token counts spanning orders of magnitude "
                                 "(zeros land in the first bin, labeled 0→1). "
                                 "linear bins: equal-width edges."),
        dcc.RadioItems(
            id="at-turns-xscale-radio",
            options=[{"label": " log bins", "value": "log"},
                     {"label": " linear bins", "value": "linear"}],
            value="log", style={"fontSize": F_SMALL},
        ),
        controls.label("Bins",
                       info_text="Number of histogram bins — more bins, finer "
                                 "detail, noisier counts."),
        dcc.Slider(id="at-turns-bins-slider", min=20, max=120, step=10, value=60,
                   marks={20: "20", 60: "60", 120: "120"}),
        html.Div(id="at-turns-gate-status",
                 style={"fontSize": "12px", "color": "#666", "marginTop": "16px",
                        "fontFamily": "monospace", "whiteSpace": "pre-wrap"}),
        html.Div("GPU type is not recorded in these traces — hardware enters on "
                 "the Deep-dive tab (trace × architecture × GPU).",
                 style={"fontSize": "11px", "color": "#999", "marginTop": "14px"}),
    ])

    graphs = [dcc.Graph(id=f"at-turns-graph-{dim}", config=GRAPH_CONFIG,
                        style={"height": "270px"})
              for dim in DIMENSIONS]
    center = controls.center_column(graphs)

    stores = [dcc.Store(id="at-turns-filter-store")]
    return html.Div(id="at-turns-tab", style={"height": "100%"},
                    children=[controls.tab_root(side, center, stores)])
