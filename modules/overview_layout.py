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
                ],
            ),
        ],
    )
