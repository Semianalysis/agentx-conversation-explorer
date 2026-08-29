"""Trace deep-dive tab layout.

The conversation list here is the SAME object as the Explorer list (identical
rows, identical checkbox state — one selection store syncs both); only the
viewport (this tab's sort/filter) is local. Charts aggregate the checked
conversations (all of them when nothing is checked).

Axes are picked in a measure matrix: measures down the left, one radio per
column for x, y1, y2, y3. The three charts share the x measure.
"""
from __future__ import annotations

from dash import dash_table, dcc, html

from modules.explorer_layout import TABLE_COLUMNS
from modules.measures import ALL_MEASURES, DEFAULT_AXES, X_MEASURES, Y_MEASURES
from modules.theme import F_SMALL, GRAPH_CONFIG, MONO

_ROW_H = "26px"


def _axis_radio(id_: str, allowed: dict, default: str) -> dcc.RadioItems:
    """One matrix column: a radio per measure row, disabled where the measure
    doesn't apply to this axis. Blank labels — the shared label column on the
    left names the rows (option counts identical everywhere = rows align)."""
    return dcc.RadioItems(
        id=id_,
        options=[{"label": "", "value": k, "disabled": k not in allowed}
                 for k in ALL_MEASURES],
        value=default,
        labelStyle={"display": "block", "height": _ROW_H, "margin": "0",
                    "textAlign": "center"},
        style={"width": "34px"},
    )


def _measure_matrix() -> html.Div:
    header_style = {"fontSize": "11px", "fontWeight": "700", "height": "20px",
                    "textAlign": "center"}
    label_col = html.Div(
        [html.Div("measure", style=dict(header_style, textAlign="left"))] + [
            html.Div(m["label"], title=m["label"],
                     style={"fontSize": "11px", "height": _ROW_H,
                            "lineHeight": _ROW_H, "whiteSpace": "nowrap",
                            "overflow": "hidden", "textOverflow": "ellipsis",
                            "color": "#333"})
            for m in ALL_MEASURES.values()
        ],
        style={"flex": "1 1 auto", "minWidth": "0"},
    )
    axis_cols = [
        html.Div([html.Div(name, style=header_style),
                  _axis_radio(f"at-deep-{name}-radio", allowed, DEFAULT_AXES[name])],
                 style={"flex": "0 0 34px"})
        for name, allowed in (("x", X_MEASURES), ("y1", Y_MEASURES),
                              ("y2", Y_MEASURES), ("y3", Y_MEASURES))
    ]
    return html.Div(
        style={"display": "flex", "gap": "2px", "padding": "8px",
               "border": "1px solid #ddd", "borderRadius": "6px",
               "background": "white", "marginTop": "8px"},
        children=[label_col] + axis_cols,
    )


def layout() -> html.Div:
    side = html.Div(
        style={"flex": "0 0 38%", "minWidth": "540px", "maxWidth": "38%",
               "display": "flex", "flexDirection": "column",
               "padding": "10px 12px", "borderRight": "1px solid #ddd",
               "background": "#fcfcfc", "overflow": "hidden"},
        children=[
            html.Div("Conversations", style={"fontSize": F_SMALL, "fontWeight": "700",
                                             "margin": "0 0 4px"}),
            html.Div("The same list as on Explorer — checks are shared both ways "
                     "(shift-click a checkbox to toggle a whole range). Charts "
                     "aggregate the checked conversations; nothing checked = all. "
                     "Sort/filter here is viewport-local.",
                     style={"fontSize": "11px", "color": "#999", "margin": "0 0 8px"}),
            html.Div(
                dash_table.DataTable(
                    id="at-deep-conv-table",
                    columns=TABLE_COLUMNS,
                    data=[],
                    sort_action="native",
                    sort_mode="multi",
                    filter_action="native",
                    row_selectable="multi",
                    selected_rows=[],
                    page_action="none",
                    fixed_rows={"headers": True},
                    style_table={"height": "100%", "overflowY": "auto",
                                 "overflowX": "auto"},
                    style_cell={"fontFamily": MONO, "fontSize": "12px",
                                "textAlign": "right", "padding": "2px 6px",
                                "minWidth": "58px", "maxWidth": "230px",
                                "whiteSpace": "nowrap", "overflow": "hidden",
                                "textOverflow": "ellipsis"},
                    style_header={"fontWeight": "700", "background": "#f2f2f2"},
                    style_filter={"background": "#fbfbf3"},
                ),
                style={"flex": "1 1 auto", "minHeight": "0", "overflow": "hidden"},
            ),
            html.Div("Axes — one x (shared by all three charts), one measure per y:",
                     style={"fontSize": "11px", "color": "#666", "marginTop": "8px"}),
            _measure_matrix(),
            dcc.Checklist(
                id="at-deep-groupopts-cl",
                options=[
                    {"label": " mean across conversations (grouped)",
                     "value": "grouped"},
                    {"label": " dotted min/max envelope", "value": "envelope"},
                ],
                value=[], style={"fontSize": "11px", "marginTop": "6px"}),
            html.Div("Grouped mode: at each x, conversations that have ended "
                     "drop out of mean/min/max — never padded with a default.",
                     style={"fontSize": "10px", "color": "#aaa", "marginTop": "2px"}),
        ],
    )

    center = html.Div(
        style={"flex": "1 1 0%", "minWidth": "0", "display": "flex",
               "flexDirection": "column", "overflow": "hidden",
               "padding": "4px 10px", "gap": "2px"},
        children=[
            dcc.Store(id="at-deep-axes-store"),
            dcc.Store(id="at-deep-sort-store"),
        ] + [
            # each chart takes exactly a third of the window height and the
            # responsive figures follow their container on window resize
            html.Div(dcc.Graph(id=f"at-deep-y{i}-graph", config=GRAPH_CONFIG,
                               style={"height": "100%", "width": "100%"}),
                     style={"flex": "1 1 0%", "minHeight": "0"})
            for i in (1, 2, 3)
        ],
    )

    right = html.Div(
        id="at-deep-summary-panel",
        style={"flex": "0 0 17%", "minWidth": "250px", "maxWidth": "17%",
               "overflowY": "auto", "padding": "12px",
               "borderLeft": "1px solid #ddd", "background": "#fcfcfc"},
    )

    return html.Div(
        id="at-deep-tab", style={"height": "100%"},
        children=[html.Div(
            style={"display": "flex", "flexDirection": "row", "height": "100%",
                   "minHeight": "0", "overflow": "hidden"},
            children=[side, center, right],
        )],
    )
