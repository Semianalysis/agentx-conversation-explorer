"""Turns tab callbacks: dataset/model options, filter coalescer, histogram render."""
from __future__ import annotations

import logging

from dash import Input, Output, State
from dash.exceptions import PreventUpdate

from modules import api_client, records
from modules.explorer_data import apply_selection
from modules.figures import empty_figure, histogram_figure
from modules.records import DIMENSIONS
from modules.theme import PALETTE
from modules.turns_data import dimension_values, filter_records

logger = logging.getLogger(__name__)

_DIM_COLORS = dict(zip(DIMENSIONS, PALETTE))


def cached_dataset_options() -> list[dict]:
    """Datasets with at least one cached conversation (shared by analysis tabs)."""
    opts = []
    try:
        datasets = api_client.fetch_datasets()
    except Exception:
        return []
    for d in datasets:
        n = len(api_client.cached_conversation_ids(d["slug"]))
        if n:
            opts.append({"label": f"{d.get('label', d['slug'])} ({n} convs cached)",
                         "value": d["slug"]})
    return opts


def register_turns_callbacks(app) -> None:
    @app.callback(
        Output("at-turns-dataset-dd", "options"),
        Input("at-tabs", "value"),
        Input("at-overview-cache-store", "data"),
    )
    def dataset_options(_tab, _cache):
        try:
            return cached_dataset_options()
        except Exception:
            logger.exception("turns dataset options failed")
            raise PreventUpdate

    @app.callback(
        Output("at-turns-models-dd", "options"),
        Input("at-turns-dataset-dd", "value"),
        prevent_initial_call=True,
    )
    def model_options(slug):
        try:
            if not slug:
                return []
            pool = records.load_records(slug)
            return [{"label": m, "value": m} for m in records.pool_models(pool)]
        except Exception:
            logger.exception("turns model options failed")
            raise PreventUpdate

    @app.callback(
        Output("at-turns-filter-store", "data"),
        Input("at-turns-dataset-dd", "value"),
        Input("at-turns-models-dd", "value"),
        Input("at-turns-roles-cl", "value"),
        Input("at-turns-xscale-radio", "value"),
        Input("at-turns-bins-slider", "value"),
        prevent_initial_call=True,
    )
    def coalesce_filters(slug, models, roles, xscale, n_bins):
        return {"slug": slug, "models": models or [], "roles": roles or [],
                "xscale": xscale, "n_bins": n_bins}

    @app.callback(
        [Output(f"at-turns-graph-{dim}", "figure") for dim in DIMENSIONS],
        Output("at-turns-gate-status", "children"),
        Input("at-turns-filter-store", "data"),
        Input("at-explorer-selection-store", "data"),
        prevent_initial_call=True,
    )
    def render(filters, selection):
        try:
            if not filters or not filters.get("slug"):
                figs = [empty_figure(DIMENSIONS[d]["label"], "pick a dataset")
                        for d in DIMENSIONS]
                return *figs, ""
            pool = records.load_records(filters["slug"])
            pool, sel_gates = apply_selection(pool, selection, filters["slug"])
            subset, gates = filter_records(
                pool, {"models": filters["models"], "roles": filters["roles"]})
            gates = {**sel_gates, **gates}
            log_x = filters["xscale"] == "log"
            figs = []
            for dim in DIMENSIONS:
                values = dimension_values(subset, dim)
                figs.append(histogram_figure(
                    values, DIMENSIONS[dim]["label"], log_x, filters["n_bins"],
                    color=_DIM_COLORS[dim], gates=gates))
            gate_txt = "rows through gates:\n" + " → ".join(
                f"{k}={v:,}" for k, v in gates.items())
            return *figs, gate_txt
        except Exception:
            logger.exception("turns render failed")
            raise PreventUpdate
