"""Correlations tab layout: three histogram charts (slots y1/y2/y3) with a
deep-dive-style measure matrix (x mode + one y measure per chart), and
color-coded bin SELECTIONS on one shared selection chart. Conditioning
always applies: every non-empty selection stacks its contribution on the
other charts. Inspector sections (one per selection, never fewer than one)
are rendered by the callbacks; 'Add selection' always sits below them; the
charts flex-fill the window height. Chart wrappers get a corr-cursor-<k>
class (assets/corr_cursors.css) so the armed inspector's colored pick-cursor
shows exactly where a click will land. All Correlations dcc.Stores live here.
"""
from __future__ import annotations

from dash import dcc, html

from modules import controls
from modules.correlations_data import (CHART_SLOTS, DEFAULT_AXES, MEASURES,
                                       initial_store)
from modules.theme import F_SMALL, GRAPH_CONFIG

_ROW_H = "24px"

_AXIS_HEADER_INFO = {
    "y1": "X measure of the top chart (its bars count requests per bin).",
    "y2": "X measure of the middle chart.",
    "y3": "X measure of the bottom chart.",
}
_COL_LABEL = {"y1": "x1", "y2": "x2", "y3": "x3"}  # slots keep internal ids


def _axis_radio(id_: str, allowed: dict, default: str) -> dcc.RadioItems:
    """One matrix column: a radio per measure row, disabled where the measure
    doesn't apply to this axis (option counts identical -> rows align)."""
    return dcc.RadioItems(
        id=id_,
        options=[{"label": "", "value": k, "disabled": k not in allowed}
                 for k in MEASURES],
        value=default,
        labelStyle={"display": "block", "height": _ROW_H, "margin": "0",
                    "textAlign": "center"},
        style={"width": "30px"},
    )


def _measure_matrix() -> html.Div:
    header_style = {"fontSize": "11px", "fontWeight": "700", "height": "20px",
                    "textAlign": "center"}
    label_col = html.Div(
        [html.Div("measure", style=dict(header_style, textAlign="left"))] + [
            html.Div(
                [html.Span(m["label"],
                           style={"flex": "1 1 auto", "minWidth": "0",
                                  "whiteSpace": "nowrap", "overflow": "hidden",
                                  "textOverflow": "ellipsis"}),
                 controls.info(m["info"])],
                title=m["info"],
                style={"fontSize": "11px", "height": _ROW_H,
                       "display": "flex", "alignItems": "center",
                       "color": "#333"})
            for m in MEASURES.values()
        ],
        style={"flex": "1 1 auto", "minWidth": "0"},
    )
    axis_cols = [
        html.Div([html.Div(_COL_LABEL[name], title=_AXIS_HEADER_INFO[name],
                           style=dict(header_style, cursor="help")),
                  _axis_radio(f"at-corr-{name}-radio", MEASURES,
                              DEFAULT_AXES[name])],
                 style={"flex": "0 0 30px"})
        for name in CHART_SLOTS
    ]
    return html.Div(
        style={"display": "flex", "gap": "2px", "padding": "6px",
               "border": "1px solid #ddd", "borderRadius": "6px",
               "background": "white", "marginTop": "4px"},
        children=[label_col] + axis_cols,
    )


def layout() -> html.Div:
    side = controls.sidebar([
        controls.label("Dataset",
                       info_text="The dataset is SHARED state: every tab is a "
                                 "viewport onto the same data, and picking a "
                                 "dataset here switches all tabs. An Explorer "
                                 "conversation selection also scopes these "
                                 "charts. Models, roles, measures, scale, and "
                                 "the selections below are viewport options — "
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
        controls.label("Axes — one x measure per chart; y = request count",
                       info_text="x1/x2/x3 pick each chart's x measure; every "
                                 "chart is a histogram counting requests per "
                                 "bin. Hover a measure name for what it "
                                 "means. Changing a measure re-bins its "
                                 "chart, so selections reset."),
        _measure_matrix(),
        controls.label("X scale",
                       info_text="log bins: geometric bin edges — right for "
                                 "values spanning orders of magnitude (zeros "
                                 "land in the first bin, labeled 0→1). "
                                 "linear bins: equal-width edges. Changing "
                                 "this re-bins the charts, so selections "
                                 "reset."),
        dcc.RadioItems(
            id="at-corr-xscale-radio",
            options=[{"label": " log bins", "value": "log"},
                     {"label": " linear bins", "value": "linear"}],
            value="log", style={"fontSize": F_SMALL},
        ),
        controls.label("Turn range (zoom)",
                       info_text="Restrict every chart to requests whose "
                                 "turn # (sequence in conversation) lies in "
                                 "[first, last]. Leave empty for open ends. "
                                 "Changing it re-bins, so selections reset."),
        html.Div(style={"display": "flex", "gap": "6px"}, children=[
            dcc.Input(id="at-corr-turnlo-input", type="number",
                      placeholder="first turn", debounce=True,
                      style={"width": "50%", "fontSize": F_SMALL}),
            dcc.Input(id="at-corr-turnhi-input", type="number",
                      placeholder="last turn", debounce=True,
                      style={"width": "50%", "fontSize": F_SMALL}),
        ]),
        html.Div("Click a bar in a histogram to add or remove from current "
                 "selection.",
                 style={"fontSize": "11px", "color": "#666", "marginTop": "14px",
                        "fontStyle": "italic"}),
        controls.label("Selections",
                       info_text="Inspectors partition the bins of ONE "
                                 "selection chart (the chart of your first "
                                 "pick) — a bin belongs to at most one "
                                 "inspector, and picking it with another "
                                 "inspector's cursor transfers it. Click an "
                                 "inspector's colored arrow to ARM it: the "
                                 "mouse cursor takes its color over the "
                                 "selection chart (over every chart while "
                                 "nothing is selected yet — the first pick "
                                 "may land anywhere; other charts keep the "
                                 "default cursor and ignore clicks). "
                                 "Conditioning always applies: each "
                                 "selection's matching requests stack on the "
                                 "OTHER charts in its color, section size = "
                                 "how much of that bar correlates. The "
                                 "color-coded Clear empties a section (it "
                                 "stays). Selections reset when the dataset, "
                                 "filters, measures, x scale, assumptions, "
                                 "or Explorer selection change."),
        html.Div(id="at-corr-selection-status",
                 style={"fontSize": "11px", "color": "#666",
                        "margin": "0 0 4px"}),
        html.Div(id="at-corr-hint",
                 style={"fontSize": "11px", "color": "#a60",
                        "whiteSpace": "pre-wrap"}),
        html.Div(id="at-corr-inspectors"),
        html.Button("Add selection", id="at-corr-newsel-btn", n_clicks=0,
                    title="Add another color-coded selection and arm it "
                          "(its cursor does the next bin picks).",
                    style={"fontSize": "12px", "padding": "3px 8px",
                           "marginTop": "8px"}),
        html.Div(id="at-corr-gate-status",
                 style={"fontSize": "12px", "color": "#666", "marginTop": "16px",
                        "fontFamily": "monospace", "whiteSpace": "pre-wrap"}),
    ])

    center = html.Div(
        style={"flex": "1 1 0%", "minWidth": "0", "display": "flex",
               "flexDirection": "column", "overflow": "hidden",
               "padding": "4px 10px", "gap": "2px"},
        children=[
            # each histogram takes a third of the window height; the wrapper
            # id carries the corr-cursor-<k> class for the armed inspector
            html.Div(dcc.Graph(id=f"at-corr-graph-{slot}", config=GRAPH_CONFIG,
                               style={"height": "100%", "width": "100%"}),
                     id=f"at-corr-wrap-{slot}",
                     style={"flex": "1 1 0%", "minHeight": "0"})
            for slot in CHART_SLOTS
        ],
    )

    stores = [
        dcc.Store(id="at-corr-filter-store"),
        dcc.Store(id="at-corr-axes-store"),
        dcc.Store(id="at-corr-selections-store", data=initial_store()),
    ]
    return html.Div(id="at-corr-tab", style={"height": "100%"},
                    children=[controls.tab_root(side, center, stores)])
