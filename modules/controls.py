"""Sidebar control factories shared by the analysis tabs."""
from __future__ import annotations

from dash import dcc, html

from modules.theme import F_SMALL, LEFT_W

NONE = "-none-"  # null sentinel for all dropdowns


def label(text: str) -> html.Div:
    return html.Div(text, style={"fontSize": F_SMALL, "fontWeight": "600",
                                 "margin": "10px 0 3px", "color": "#333"})


def dropdown(id_: str, placeholder: str, multi: bool = False, **kwargs) -> dcc.Dropdown:
    return dcc.Dropdown(id=id_, placeholder=placeholder, multi=multi,
                        style={"fontSize": F_SMALL}, **kwargs)


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
