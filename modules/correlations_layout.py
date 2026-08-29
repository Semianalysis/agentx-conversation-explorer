"""Correlations tab layout: build color-coded bin SELECTIONS on one histogram
each; each selection's own apply toggle conditions the other histograms on
its matched requests. Inspector sections (one per selection, never fewer
than one) live in the left panel and are rendered by the callbacks; 'Add
selection' always sits below them; the three histograms flex-fill the window
height. All Correlations dcc.Stores live here.
"""
from __future__ import annotations

from dash import dcc, html

from modules import controls
from modules.correlations_data import initial_store
from modules.records import DIMENSIONS
from modules.theme import F_SMALL, GRAPH_CONFIG

_BTN_STYLE = {"fontSize": "12px", "padding": "3px 8px"}


def layout() -> html.Div:
    side = controls.sidebar([
        controls.label("Dataset",
                       info_text="The dataset is SHARED state: every tab is a "
                                 "viewport onto the same data, and picking a "
                                 "dataset here switches all tabs. An Explorer "
                                 "conversation selection also scopes these "
                                 "charts. Models, roles, scale, and the "
                                 "selections below are viewport options — "
                                 "local to this tab."),
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
        controls.label("X scale",
                       info_text="log bins: geometric bin edges — right for "
                                 "token counts spanning orders of magnitude "
                                 "(zeros land in the first bin, labeled 0→1). "
                                 "linear bins: equal-width edges. Changing "
                                 "this re-bins the charts, so selections "
                                 "reset (bin positions would otherwise refer "
                                 "to different token ranges)."),
        dcc.RadioItems(
            id="at-corr-xscale-radio",
            options=[{"label": " log bins", "value": "log"},
                     {"label": " linear bins", "value": "linear"}],
            value="log", style={"fontSize": F_SMALL},
        ),
        html.Div("Click a bar in a histogram to add or remove from current "
                 "selection.",
                 style={"fontSize": "11px", "color": "#666", "marginTop": "14px",
                        "fontStyle": "italic"}),
        controls.label("Selections",
                       info_text="Click bins (or box-select) on ONE histogram "
                                 "to build the live selection — that chart "
                                 "becomes its controlling histogram; clicking "
                                 "a picked bin again removes it. Each "
                                 "selection below shows per-bin detail and a "
                                 "clickable strip to add/remove bins. Its "
                                 "'apply' toggle draws its matching requests "
                                 "on the OTHER histograms (stacked, in the "
                                 "selection's color) over the gray full "
                                 "distribution; its color-coded Clear button "
                                 "empties it (the section stays). 'Add "
                                 "selection' below starts another color for "
                                 "side-by-side comparison. Selections reset "
                                 "when the dataset, filters, x scale, or "
                                 "Explorer selection change."),
        html.Div(id="at-corr-selection-status",
                 style={"fontSize": "11px", "color": "#666",
                        "margin": "0 0 4px"}),
        html.Div(id="at-corr-hint",
                 style={"fontSize": "11px", "color": "#a60",
                        "whiteSpace": "pre-wrap"}),
        html.Div(id="at-corr-inspectors"),
        html.Button("Add selection", id="at-corr-newsel-btn", n_clicks=0,
                    title="Add another color-coded selection and make it "
                          "live (bin clicks edit the live selection).",
                    style={**_BTN_STYLE, "marginTop": "8px"}),
        html.Div(id="at-corr-gate-status",
                 style={"fontSize": "12px", "color": "#666", "marginTop": "16px",
                        "fontFamily": "monospace", "whiteSpace": "pre-wrap"}),
    ])

    center = html.Div(
        style={"flex": "1 1 0%", "minWidth": "0", "display": "flex",
               "flexDirection": "column", "overflow": "hidden",
               "padding": "4px 10px", "gap": "2px"},
        children=[
            # each histogram takes a third of the window height; responsive
            # figures follow their container on resize
            html.Div(dcc.Graph(id=f"at-corr-graph-{dim}", config=GRAPH_CONFIG,
                               style={"height": "100%", "width": "100%"}),
                     style={"flex": "1 1 0%", "minHeight": "0"})
            for dim in DIMENSIONS
        ],
    )

    stores = [
        dcc.Store(id="at-corr-filter-store"),
        dcc.Store(id="at-corr-selections-store", data=initial_store()),
    ]
    return html.Div(id="at-corr-tab", style={"height": "100%"},
                    children=[controls.tab_root(side, center, stores)])
