"""Overview tab layout: import bar + dataset summary cards.

All Overview dcc.Stores live here. The datasets store is pre-populated from the
disk cache at app start (no network at import time — cache reads only).
"""
from __future__ import annotations

import logging

from dash import dcc, html

from modules import api_client
from modules.theme import F_BASE, F_SMALL

logger = logging.getLogger(__name__)


def _initial_datasets() -> list[dict]:
    """Disk-cache-only bootstrap; empty list if nothing imported yet."""
    try:
        slugs = [d["slug"] for d in api_client.fetch_datasets()] if (
            api_client.DATA_DIR / "datasets.json").is_file() else []
        return [api_client.fetch_dataset_detail(s) for s in slugs
                if (api_client.DATA_DIR / s / "detail.json").is_file()]
    except Exception:
        logger.exception("initial dataset cache read failed")
        return []


def _initial_cache_counts() -> dict:
    return {d["slug"]: len(api_client.cached_conversation_ids(d["slug"]))
            for d in _initial_datasets()}


def _finder_section() -> html.Div:
    """Search for a dataset - published ones on the website, or exports
    already cached on this machine - preview its stats, and load it into
    this session. Nothing is loaded by browsing: the preview reads the
    dataset card only, and the Load button is what adds it."""
    from modules.controls import info
    return html.Div(
        style={"marginTop": "6px", "marginBottom": "18px",
               "border": "1px solid #ddd", "borderRadius": "8px",
               "padding": "12px", "background": "white"},
        children=[
            html.Div(["Find a dataset",
                      info("Search the published AgentX datasets, or the "
                           "exports already imported on this machine. "
                           "Previewing shows the dataset's own summary "
                           "stats; nothing enters the session until you "
                           "press Load, which is what makes it appear in "
                           "the dataset dropdowns.")],
                     style={"fontWeight": "700", "fontSize": F_BASE,
                            "display": "flex", "alignItems": "center"}),
            html.Div(style={"display": "flex", "gap": "10px",
                            "alignItems": "center", "margin": "8px 0",
                            "flexWrap": "wrap"},
                     children=[
                         dcc.RadioItems(
                             id="at-summary-scope-radio",
                             options=[{"label": " online (published)",
                                       "value": "online"},
                                      {"label": " local (on this machine)",
                                       "value": "local"}],
                             value="online", inline=True,
                             labelStyle={"marginRight": "12px"},
                             style={"fontSize": F_SMALL}),
                         dcc.Input(id="at-summary-search-input", type="text",
                                   debounce=True, placeholder="name contains…",
                                   style={"width": "260px",
                                          "fontSize": F_SMALL}),
                         html.Button("Search", id="at-summary-search-btn",
                                     n_clicks=0,
                                     style={"fontSize": F_SMALL,
                                            "padding": "4px 10px"}),
                         html.Button("Load into session",
                                     id="at-summary-load-btn", n_clicks=0,
                                     title="Add the previewed dataset to this "
                                           "session's sources so it appears "
                                           "in the dataset dropdowns.",
                                     style={"fontSize": F_SMALL,
                                            "padding": "4px 10px"}),
                     ]),
            dcc.Store(id="at-summary-found-store"),
            html.Div(id="at-summary-found-list"),
            html.Div(id="at-summary-preview"),
        ],
    )


def _internal_section() -> html.Div:
    """Internal (unpublished) sources — rendered ONLY in internal mode
    (AGENTX_INTERNAL=1). The public build shows nothing: no names, no
    controls; listings are fetched live from HF, and private datasets appear
    only if the user's own HF_TOKEN can read them."""
    from modules import hf_client
    from modules.controls import info
    if not hf_client.internal_mode():
        return html.Div()  # public build: no internal components at all
    return html.Div(
        style={"marginTop": "22px", "borderTop": "2px dashed #ca8",
               "paddingTop": "10px"},
        children=[
            html.Div(["Internal sources — unpublished datasets",
                      info("Visible because AGENTX_INTERNAL=1. Lists every "
                           "dataset of the configured HF namespaces "
                           "(AGENTX_HF_SOURCES, default semianalysisai) — "
                           "with an HF_TOKEN set, private datasets your "
                           "token can read appear too, so access follows "
                           "HuggingFace's own permissions, per user. "
                           "Importing flattens the raw traces locally into "
                           "the same format as published datasets; every "
                           "tab can then explore them. Nothing here exists "
                           "in the public build.")],
                     style={"fontWeight": "700", "fontSize": F_BASE,
                            "display": "flex", "alignItems": "center"}),
            html.Div(style={"display": "flex", "alignItems": "center",
                            "gap": "12px", "margin": "8px 0"},
                     children=[
                         html.Button("List internal datasets",
                                     id="at-summary-hf-list-btn", n_clicks=0,
                                     style={"fontSize": F_SMALL,
                                            "padding": "4px 10px"}),
                         dcc.Loading(html.Div(id="at-summary-hf-status",
                                              style={"fontSize": F_SMALL,
                                                     "color": "#555"}),
                                     type="dot"),
                     ]),
            # the checklist exists in the INITIAL layout (empty) — the Load
            # callback takes State from it, and Dash rejects callbacks that
            # reference ids missing from the layout (no
            # suppress_callback_exceptions here, deliberately)
            dcc.Checklist(id="at-summary-hf-select-cl", options=[], value=[],
                          labelStyle={"display": "block",
                                      "fontFamily": "Consolas, monospace",
                                      "fontSize": "12px"}),
            html.Div(id="at-summary-hf-unsupported"),
            html.Div(style={"display": "flex", "alignItems": "center",
                            "gap": "12px", "marginTop": "8px"},
                     children=[
                         html.Button("Load selected",
                                     id="at-summary-hf-load-btn", n_clicks=0,
                                     title="Import every checked dataset in "
                                           "the background (raw traces from "
                                           "HuggingFace; can take minutes "
                                           "per dataset).",
                                     style={"fontSize": F_SMALL,
                                            "padding": "4px 10px"}),
                         html.Div(id="at-summary-hf-progress"),
                     ]),
            dcc.Interval(id="at-summary-hf-interval", interval=1000),
        ],
    )


def layout() -> html.Div:
    return html.Div(
        id="at-summary-tab",
        style={"display": "flex", "flexDirection": "column", "height": "100%",
               "overflow": "hidden"},
        children=[
            dcc.Store(id="at-summary-datasets-store", data=_initial_datasets()),
            dcc.Store(id="at-summary-cache-store", data=_initial_cache_counts()),
            html.Div(
                style={"display": "flex", "alignItems": "center", "gap": "12px",
                       "padding": "10px 14px", "borderBottom": "1px solid #ddd",
                       "flex": "0 0 auto"},
                children=[
                    html.Button("Refresh published datasets",
                                id="at-summary-import-btn",
                                n_clicks=0,
                                title="Fetch the dataset list and summary "
                                      "stats from the AgentX API into the "
                                      "local data/ cache. Per-conversation "
                                      "traces are downloaded separately with "
                                      "each dataset card's button.",
                                style={"fontSize": F_BASE, "padding": "6px 14px"}),
                    dcc.Loading(html.Div(id="at-summary-import-status",
                                         style={"fontSize": F_SMALL, "color": "#555"}),
                                type="dot"),
                    html.Div("Source: inferencex.semianalysis.com/api/v1 → local cache in data/",
                             style={"fontSize": F_SMALL, "color": "#999",
                                    "marginLeft": "auto"}),
                ],
            ),
            html.Div(
                style={"flex": "1 1 auto", "overflowY": "auto", "padding": "14px"},
                children=[
                    _finder_section(),
                    html.Div(id="at-summary-loaded-title",
                             style={"fontWeight": "700", "fontSize": F_BASE,
                                    "margin": "4px 0 8px"}),
                    html.Div(id="at-summary-selection-summary"),
                    html.Div(
                        id="at-summary-cards",
                        style={"display": "flex", "flexWrap": "wrap", "gap": "14px",
                               "alignItems": "flex-start"},
                    ),
                    _internal_section(),
                ],
            ),
        ],
    )
