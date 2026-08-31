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


def _internal_section() -> html.Div:
    """Internal (unpublished) sources — rendered ONLY in internal mode
    (AGENTX_INTERNAL=1). The public build shows nothing: no names, no
    controls; listings are fetched live from HF, and private datasets appear
    only if the user's own HF_TOKEN can read them."""
    from modules import hf_client
    from modules.controls import info
    if not hf_client.internal_mode():
        return html.Div(id="at-overview-hf-list", style={"display": "none"})
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
                                     id="at-overview-hf-list-btn", n_clicks=0,
                                     style={"fontSize": F_SMALL,
                                            "padding": "4px 10px"}),
                         dcc.Loading(html.Div(id="at-overview-hf-status",
                                              style={"fontSize": F_SMALL,
                                                     "color": "#555"}),
                                     type="dot"),
                     ]),
            html.Div(id="at-overview-hf-list"),
            html.Div(style={"display": "flex", "alignItems": "center",
                            "gap": "12px", "marginTop": "8px"},
                     children=[
                         html.Button("Load selected",
                                     id="at-overview-hf-load-btn", n_clicks=0,
                                     title="Import every checked dataset in "
                                           "the background (raw traces from "
                                           "HuggingFace; can take minutes "
                                           "per dataset).",
                                     style={"fontSize": F_SMALL,
                                            "padding": "4px 10px"}),
                         html.Div(id="at-overview-hf-progress"),
                     ]),
            dcc.Interval(id="at-overview-hf-interval", interval=1000),
        ],
    )


def layout() -> html.Div:
    return html.Div(
        id="at-overview-tab",
        style={"display": "flex", "flexDirection": "column", "height": "100%",
               "overflow": "hidden"},
        children=[
            dcc.Store(id="at-overview-datasets-store", data=_initial_datasets()),
            dcc.Store(id="at-overview-cache-store", data=_initial_cache_counts()),
            html.Div(
                style={"display": "flex", "alignItems": "center", "gap": "12px",
                       "padding": "10px 14px", "borderBottom": "1px solid #ddd",
                       "flex": "0 0 auto"},
                children=[
                    html.Button("Import / refresh datasets", id="at-overview-import-btn",
                                n_clicks=0,
                                title="Fetch the dataset list and summary "
                                      "stats from the AgentX API into the "
                                      "local data/ cache. Per-conversation "
                                      "traces are downloaded separately with "
                                      "each dataset card's button.",
                                style={"fontSize": F_BASE, "padding": "6px 14px"}),
                    dcc.Loading(html.Div(id="at-overview-import-status",
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
                    html.Div(id="at-overview-selection-summary"),
                    html.Div(
                        id="at-overview-cards",
                        style={"display": "flex", "flexWrap": "wrap", "gap": "14px",
                               "alignItems": "flex-start"},
                    ),
                    _internal_section(),
                ],
            ),
        ],
    )
