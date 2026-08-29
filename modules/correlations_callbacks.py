"""Correlations tab callbacks.

Condition-range picking is one circular callback (Dash supports self-loop
outputs within a single callback): lo/hi inputs, dim dropdown, clear button,
and box-select events on any of the three histograms all converge on
at-corr-condition-store + the synced widget values.
"""
from __future__ import annotations

import logging

from dash import Input, Output, State, callback_context, no_update
from dash.exceptions import PreventUpdate

from modules import records
from modules.correlations_data import conditional_subset
from modules.explorer_data import apply_selection
from modules.figures import empty_figure, histogram_figure, selection_to_token_range
from modules.records import DIMENSIONS
from modules.theme import PALETTE
from modules.turns_callbacks import cached_dataset_options
from modules.turns_data import dimension_values, filter_records

logger = logging.getLogger(__name__)

_DIM_COLORS = dict(zip(DIMENSIONS, PALETTE))
_N_BINS = 60


def register_correlations_callbacks(app) -> None:
    @app.callback(
        Output("at-corr-dataset-dd", "options"),
        Input("at-tabs", "value"),
        Input("at-overview-cache-store", "data"),
    )
    def dataset_options(_tab, _cache):
        try:
            return cached_dataset_options()
        except Exception:
            logger.exception("corr dataset options failed")
            raise PreventUpdate

    @app.callback(
        Output("at-corr-models-dd", "options"),
        Input("at-corr-dataset-dd", "value"),
        prevent_initial_call=True,
    )
    def model_options(slug):
        try:
            if not slug:
                return []
            pool = records.load_records(slug)
            return [{"label": m, "value": m} for m in records.pool_models(pool)]
        except Exception:
            logger.exception("corr model options failed")
            raise PreventUpdate

    @app.callback(
        Output("at-corr-filter-store", "data"),
        Input("at-corr-dataset-dd", "value"),
        Input("at-corr-models-dd", "value"),
        Input("at-corr-roles-cl", "value"),
        Input("at-corr-xscale-radio", "value"),
        prevent_initial_call=True,
    )
    def coalesce_filters(slug, models, roles, xscale):
        return {"slug": slug, "models": models or [], "roles": roles or [],
                "xscale": xscale}

    @app.callback(
        Output("at-corr-condition-store", "data"),
        Output("at-corr-cond-dim-dd", "value"),
        Output("at-corr-lo-input", "value"),
        Output("at-corr-hi-input", "value"),
        Input("at-corr-cond-dim-dd", "value"),
        Input("at-corr-lo-input", "value"),
        Input("at-corr-hi-input", "value"),
        Input("at-corr-clear-btn", "n_clicks"),
        [Input(f"at-corr-graph-{dim}", "selectedData") for dim in DIMENSIONS],
        State("at-corr-filter-store", "data"),
        State("at-explorer-selection-store", "data"),
        prevent_initial_call=True,
    )
    def pick_condition(dim_value, lo_value, hi_value, _clear_clicks, *args):
        try:
            selected_by_dim = dict(zip(DIMENSIONS, args[:len(DIMENSIONS)]))
            filters, conv_selection = args[len(DIMENSIONS)], args[len(DIMENSIONS) + 1]
            trig = callback_context.triggered_id

            if trig == "at-corr-clear-btn":
                return {"dim": dim_value, "lo": None, "hi": None}, no_update, None, None

            if isinstance(trig, str) and trig.startswith("at-corr-graph-"):
                dim = trig.removeprefix("at-corr-graph-")
                sel = selected_by_dim.get(dim)
                if not sel or "range" not in sel or not filters or not filters.get("slug"):
                    raise PreventUpdate
                pool = records.load_records(filters["slug"])
                pool, _ = apply_selection(pool, conv_selection, filters["slug"])
                subset, _ = filter_records(
                    pool, {"models": filters["models"], "roles": filters["roles"]})
                pool_values = dimension_values(subset, dim)
                if not pool_values:
                    raise PreventUpdate
                lo_tok, hi_tok = selection_to_token_range(
                    tuple(sel["range"]["x"]), pool_values, _N_BINS,
                    filters["xscale"] == "log")
                lo_tok, hi_tok = round(lo_tok), round(hi_tok)
                return ({"dim": dim, "lo": lo_tok, "hi": hi_tok},
                        dim, lo_tok, hi_tok)

            # dim dropdown or manual lo/hi edit
            return ({"dim": dim_value, "lo": lo_value, "hi": hi_value},
                    no_update, no_update, no_update)
        except PreventUpdate:
            raise
        except Exception:
            logger.exception("condition pick failed")
            raise PreventUpdate

    @app.callback(
        [Output(f"at-corr-graph-{dim}", "figure") for dim in DIMENSIONS],
        Output("at-corr-gate-status", "children"),
        Input("at-corr-filter-store", "data"),
        Input("at-corr-condition-store", "data"),
        Input("at-explorer-selection-store", "data"),
        prevent_initial_call=True,
    )
    def render(filters, condition, conv_selection):
        try:
            if not filters or not filters.get("slug"):
                figs = [empty_figure(DIMENSIONS[d]["label"], "pick a dataset")
                        for d in DIMENSIONS]
                return *figs, ""
            pool = records.load_records(filters["slug"])
            pool, sel_gates = apply_selection(pool, conv_selection, filters["slug"])
            filtered, gates = filter_records(
                pool, {"models": filters["models"], "roles": filters["roles"]})
            gates = {**sel_gates, **gates}
            log_x = filters["xscale"] == "log"

            cond = condition or {}
            cond_dim = cond.get("dim") or "context"
            lo, hi = cond.get("lo"), cond.get("hi")
            has_range = lo is not None or hi is not None
            if has_range:
                subset, cgates = conditional_subset(filtered, cond_dim, lo, hi)
                gates.update(cgates)
            else:
                subset = filtered

            figs = []
            for dim in DIMENSIONS:
                pool_values = dimension_values(filtered, dim)
                if dim == cond_dim:
                    figs.append(histogram_figure(
                        pool_values,
                        f"CONDITION — {DIMENSIONS[dim]['label']}",
                        log_x, _N_BINS, color="#d62728", gates=gates,
                        highlight_range=(lo if lo is not None else 0,
                                         hi if hi is not None else float("inf"))
                        if has_range else None,
                    ))
                else:
                    subset_values = dimension_values(subset, dim)
                    scale = (len(subset_values) / len(pool_values)) if pool_values else 1.0
                    figs.append(histogram_figure(
                        subset_values,
                        DIMENSIONS[dim]["label"]
                        + (" | condition" if has_range else " (no condition set)"),
                        log_x, _N_BINS, color=_DIM_COLORS[dim], gates=gates,
                        edge_values=pool_values,
                        overlay_values=pool_values if has_range else None,
                        overlay_scale=scale,
                    ))
            gate_txt = "rows through gates:\n" + " → ".join(
                f"{k}={v:,}" for k, v in gates.items())
            if has_range:
                gate_txt += f"\ncondition: {cond_dim} ∈ [{lo or 0}, {hi if hi is not None else '∞'}]"
            return *figs, gate_txt
        except Exception:
            logger.exception("correlations render failed")
            raise PreventUpdate
