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

from modules import controls
from modules.explorer_layout import TABLE_COLUMNS, TABLE_TOOLTIP_KWARGS
from modules.measures import ALL_MEASURES, DEFAULT_AXES, X_MEASURES, Y_MEASURES
from modules.theme import F_SMALL, GRAPH_CONFIG, MONO

_ROW_H = "26px"

_AXIS_HEADER_INFO = {
    "x": "Horizontal axis of ALL THREE charts — shared measure AND locked to "
         "one identical range, so the charts align vertically.",
    "y1": "Vertical axis of the top chart.",
    "y2": "Vertical axis of the middle chart.",
    "y3": "Vertical axis of the bottom chart.",
}


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
            for m in ALL_MEASURES.values()
        ],
        style={"flex": "1 1 auto", "minWidth": "0"},
    )
    axis_cols = [
        html.Div([html.Div(name, title=_AXIS_HEADER_INFO[name],
                           style=dict(header_style, cursor="help")),
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
            html.Div(["Conversations",
                      controls.info(
                          "Tick checkboxes to pick which conversations the "
                          "charts aggregate (nothing ticked = all of them). "
                          "Shift-click a checkbox to toggle the whole range "
                          "since your last click. This is the SAME list as on "
                          "the Explorer tab — checks carry over both ways; "
                          "only sorting and the filter row are local to this "
                          "tab. Hover a column header for what it means.")],
                     style={"fontSize": F_SMALL, "fontWeight": "700",
                            "margin": "0 0 4px", "display": "flex",
                            "alignItems": "center"}),
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
                    **TABLE_TOOLTIP_KWARGS,
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
            html.Div(["Axes — one x (shared by all three charts), one measure per y:",
                      controls.info(
                          "Pick one measure per column: x sets the horizontal "
                          "axis shared (and range-locked) by all three charts; "
                          "y1/y2/y3 set each chart's vertical axis. Hover a "
                          "measure name for what it means. Ordinals draw "
                          "linear; magnitudes draw log with zeros shown at 1.")],
                     style={"fontSize": "11px", "color": "#666", "marginTop": "8px",
                            "display": "flex", "alignItems": "center"}),
            _measure_matrix(),
            html.Div(
                style={"display": "flex", "alignItems": "center", "gap": "8px",
                       "marginTop": "6px", "fontSize": "11px"},
                children=[
                    html.Span(["x scale",
                               controls.info(
                                   "Axis scale of the shared x measure on all "
                                   "three charts. auto = the measure's natural "
                                   "scale (ordinals linear, magnitudes log). "
                                   "The locked shared range follows this "
                                   "choice.")],
                              style={"color": "#666", "fontWeight": "600",
                                     "display": "flex",
                                     "alignItems": "center"}),
                    dcc.RadioItems(
                        id="at-deep-xscale-radio",
                        options=[{"label": " auto", "value": "auto"},
                                 {"label": " linear", "value": "linear"},
                                 {"label": " log", "value": "log"}],
                        value="auto", inline=True,
                        labelStyle={"marginRight": "10px"},
                        style={"fontSize": "11px"}),
                ]),
            dcc.Checklist(
                id="at-deep-groupopts-cl",
                options=[
                    {"label": html.Span([
                        " mean across conversations (grouped)",
                        controls.info(
                            "Replace per-request markers with the MEAN across "
                            "the selected conversations at each x position. A "
                            "conversation only contributes while it still has "
                            "requests — once it ends it drops out of the "
                            "stats, never padded with defaults.")],
                        style={"display": "inline"}),
                     "value": "grouped"},
                    {"label": html.Span([
                        " dotted min/max envelope",
                        controls.info(
                            "With grouped mode on, adds dotted lines tracking "
                            "the lowest and highest single-conversation value "
                            "at each x position.")],
                        style={"display": "inline"}),
                     "value": "envelope"},
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
