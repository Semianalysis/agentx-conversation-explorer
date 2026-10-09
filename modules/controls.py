"""Sidebar control factories shared by the analysis tabs."""
from __future__ import annotations

from dash import dcc, html

from modules.theme import F_SMALL, LEFT_W

NONE = "-none-"  # null sentinel for all dropdowns


def info(text: str) -> html.Span:
    """Hoverable ⓘ tag — native browser tooltip via title=, so it works inside
    any layout (labels, table notes, checklist options) with no extra deps.
    Use for jargon and 'what can I do here' help that shouldn't fill the screen."""
    return html.Span("ⓘ", title=text, style={
        "cursor": "help", "color": "#4a90d9", "marginLeft": "5px",
        "fontSize": "11px", "fontWeight": "400", "flex": "0 0 auto",
        "userSelect": "none"})


def label(text: str, info_text: str | None = None) -> html.Div:
    children: list = [text] if info_text is None else [text, info(info_text)]
    return html.Div(children, style={"fontSize": F_SMALL, "fontWeight": "600",
                                     "margin": "10px 0 3px", "color": "#333"})


def dropdown(id_: str, placeholder: str, multi: bool = False, **kwargs) -> dcc.Dropdown:
    return dcc.Dropdown(id=id_, placeholder=placeholder, multi=multi,
                        style={"fontSize": F_SMALL}, **kwargs)


def card(title: str, rows: list[tuple], info_text: str | None = None) -> html.Div:
    """Right-column summary card. rows are (key, value) or
    (key, value, hover-help)."""
    from modules.theme import MONO
    title_children: list = [title]
    if info_text:
        title_children.append(info(info_text))
    row_divs = []
    for row in rows:
        k, v = row[0], row[1]
        tip = row[2] if len(row) > 2 else None
        row_divs.append(html.Div(
            style={"display": "flex", "justifyContent": "space-between",
                   "fontSize": "12px", "gap": "10px"},
            children=[html.Span([k, info(tip)] if tip else k,
                                style={"color": "#666"}),
                      html.Span(v, style={"fontFamily": MONO})]))
    return html.Div(
        style={"border": "1px solid #ddd", "borderRadius": "6px",
               "padding": "10px", "marginBottom": "10px", "background": "white"},
        children=[html.Div(title_children,
                           style={"fontWeight": "600", "fontSize": F_SMALL,
                                  "marginBottom": "6px", "display": "flex",
                                  "alignItems": "center"})] + row_divs,
    )


def sidebar(children: list) -> html.Div:
    return html.Div(
        style={"flex": f"0 0 {LEFT_W}", "minWidth": "0", "maxWidth": LEFT_W,
               "overflowY": "auto", "padding": "10px 12px",
               "borderRight": "1px solid #ddd", "background": "#fcfcfc"},
        children=children,
    )


def center_column(children: list, id_: str | None = None) -> html.Div:
    kwargs = {"id": id_} if id_ else {}
    return html.Div(
        style={"flex": "1 1 0%", "minWidth": "0", "overflowY": "auto",
               "padding": "8px 12px"},
        children=children, **kwargs,
    )


def tab_root(sidebar_div: html.Div, center_div: html.Div,
             stores: list, right_div: html.Div | None = None) -> html.Div:
    row_children = [sidebar_div, center_div]
    if right_div is not None:
        row_children.append(right_div)
    return html.Div(
        style={"display": "flex", "flexDirection": "row", "height": "100%",
               "minHeight": "0", "overflow": "hidden"},
        children=stores + row_children,
    )
