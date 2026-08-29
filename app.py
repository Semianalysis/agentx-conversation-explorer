"""AgentX Conversation Explorer (GitHub: AgentX Data Explorer) — standalone Dash app.

Run:  python app.py   ->  http://localhost:8050
"""
from __future__ import annotations

import logging
import sys

from dash import Dash, Input, Output, dcc, html

from modules import (correlations_callbacks, correlations_layout,
                     deepdive_callbacks, deepdive_layout, explorer_callbacks,
                     explorer_layout, overview_callbacks, overview_layout)
from modules.theme import F_BASE
from modules.version import APP_BUILD

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")

_TABS = [
    ("overview", "Overview"),
    ("explorer", "Explorer"),
    ("correlations", "Correlations"),
    ("deepdive", "Deep-dive"),
]

_APP_NAME = f"AgentX Conversation Explorer bld {APP_BUILD}"

app = Dash(__name__, title=_APP_NAME)

# All tab bodies stay MOUNTED (dcc.Tabs swaps children out, which would
# destroy each tab's stores); a style callback toggles visibility instead.
app.layout = html.Div(
    style={"display": "flex", "flexDirection": "column", "height": "100vh",
           "overflow": "hidden", "fontFamily": "Segoe UI, system-ui, sans-serif"},
    children=[
        # assets/sort_ctrl.js records the ctrl-key state of every table click
        # here, just before DataTable updates sort_by (see _register_sort_semantics)
        dcc.Input(id="at-sort-ctrl-input", type="text", value="0",
                  style={"display": "none"}),
        html.Div(
            style={"display": "flex", "alignItems": "center", "gap": "18px",
                   "padding": "8px 14px 0", "borderBottom": "1px solid #ccc",
                   "flex": "0 0 auto"},
            children=[
                html.Div(_APP_NAME,
                         style={"fontWeight": "700", "fontSize": "17px",
                                "whiteSpace": "nowrap"}),
                dcc.Tabs(
                    id="at-tabs", value="overview",
                    style={"height": "36px", "width": "780px"},
                    children=[dcc.Tab(label=lbl, value=val,
                                      style={"padding": "8px", "fontSize": F_BASE},
                                      selected_style={"padding": "8px",
                                                      "fontSize": F_BASE,
                                                      "fontWeight": "600"})
                              for val, lbl in _TABS],
                ),
            ],
        ),
        html.Div(overview_layout.layout(), id="at-tabwrap-overview",
                 style={"flex": "1 1 auto", "minHeight": "0"}),
        html.Div(explorer_layout.layout(), id="at-tabwrap-explorer",
                 style={"flex": "1 1 auto", "minHeight": "0", "display": "none"}),
        html.Div(correlations_layout.layout(), id="at-tabwrap-correlations",
                 style={"flex": "1 1 auto", "minHeight": "0", "display": "none"}),
        html.Div(deepdive_layout.layout(), id="at-tabwrap-deepdive",
                 style={"flex": "1 1 auto", "minHeight": "0", "display": "none"}),
    ],
)


@app.callback(
    [Output(f"at-tabwrap-{val}", "style") for val, _ in _TABS],
    Input("at-tabs", "value"),
)
def switch_tab(active):
    styles = []
    for val, _ in _TABS:
        style = {"flex": "1 1 auto", "minHeight": "0"}
        if val != active:
            style["display"] = "none"
        styles.append(style)
    return styles


overview_callbacks.register_overview_callbacks(app)
explorer_callbacks.register_explorer_callbacks(app)
correlations_callbacks.register_correlations_callbacks(app)
deepdive_callbacks.register_deepdive_callbacks(app)


if __name__ == "__main__":
    debug = "--debug" in sys.argv
    app.run(debug=debug, port=8050)
