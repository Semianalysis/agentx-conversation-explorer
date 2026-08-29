"""Correlations tab layout: condition on a range of one dimension, see the
conditional histograms of the others. All Correlations dcc.Stores live here.
"""
from __future__ import annotations

from dash import dcc, html

from modules import controls
from modules.theme import F_SMALL, GRAPH_CONFIG
from modules.records import DIMENSIONS


def layout() -> html.Div:
    side = controls.sidebar([
        controls.label("Dataset",
                       info_text="Pick which cached dataset to analyze. "
                                 "Datasets are imported on the Overview tab. "
                                 "An Explorer selection scopes these charts "
                                 "to the selected conversations."),
        controls.dropdown("at-corr-dataset-dd", "pick a cached dataset"),
        controls.label("Models (empty = all)",
                       info_text="Restrict to requests served by specific "
                                 "models. Empty = every model in the dataset."),
        controls.dropdown("at-corr-models-dd", "all models", multi=True),
        controls.label("Role",
                       info_text="main agent = the top-level conversation's "
                                 "own requests; subagents = nested agents "
                                 "spawned by tool calls, each with its own "
                                 "context."),
        dcc.Checklist(
            id="at-corr-roles-cl",
            options=[{"label": " main agent", "value": "main"},
                     {"label": " subagents", "value": "subagent"}],
            value=["main", "subagent"],
            style={"fontSize": F_SMALL},
        ),
        controls.label("Condition dimension",
                       info_text="The dimension you slice on. Give it a range "
                                 "below (or box-select on its histogram) and "
                                 "the other two histograms redraw showing "
                                 "ONLY the requests inside that range — the "
                                 "conditional distributions."),
        controls.dropdown(
            "at-corr-cond-dim-dd", "dimension to condition on",
            options=[{"label": DIMENSIONS[d]["label"], "value": d} for d in DIMENSIONS],
            value="context", clearable=False),
        controls.label("Condition range (tokens)",
                       info_text="Token bounds of the conditioning slice. "
                                 "Leave a side empty for open-ended. The "
                                 "faint gray outline on the other histograms "
                                 "is the unconditioned shape, rescaled, for "
                                 "comparison."),
        html.Div(style={"display": "flex", "gap": "6px"}, children=[
            dcc.Input(id="at-corr-lo-input", type="number", placeholder="min",
                      debounce=True, style={"width": "50%", "fontSize": F_SMALL}),
            dcc.Input(id="at-corr-hi-input", type="number", placeholder="max",
                      debounce=True, style={"width": "50%", "fontSize": F_SMALL}),
        ]),
        html.Div("…or box-select a range directly on the condition histogram.",
                 style={"fontSize": "11px", "color": "#999", "margin": "4px 0"}),
        html.Button("Clear range", id="at-corr-clear-btn", n_clicks=0,
                    title="Drop the condition — all three histograms return "
                          "to the unconditioned distributions.",
                    style={"fontSize": F_SMALL, "marginTop": "4px"}),
        controls.label("X scale",
                       info_text="log bins: geometric bin edges — right for "
                                 "token counts spanning orders of magnitude "
                                 "(zeros land in the first bin, labeled 0→1). "
                                 "linear bins: equal-width edges."),
        dcc.RadioItems(
            id="at-corr-xscale-radio",
            options=[{"label": " log bins", "value": "log"},
                     {"label": " linear bins", "value": "linear"}],
            value="log", style={"fontSize": F_SMALL},
        ),
        html.Div(id="at-corr-gate-status",
                 style={"fontSize": "12px", "color": "#666", "marginTop": "16px",
                        "fontFamily": "monospace", "whiteSpace": "pre-wrap"}),
    ])

    graphs = [dcc.Graph(id=f"at-corr-graph-{dim}", config=GRAPH_CONFIG,
                        style={"height": "270px"})
              for dim in DIMENSIONS]
    center = controls.center_column(graphs)

    stores = [
        dcc.Store(id="at-corr-filter-store"),
        # {'dim': ..., 'lo': tokens|None, 'hi': tokens|None}
        dcc.Store(id="at-corr-condition-store"),
    ]
    return html.Div(id="at-corr-tab", style={"height": "100%"},
                    children=[controls.tab_root(side, center, stores)])
