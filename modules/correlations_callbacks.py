"""Correlations tab callbacks.

One MUTATION callback owns the selections-store: bin clicks and box-selects
on the three histograms, clicks on inspector strips / headers / delete
buttons, and the New/Apply/Clear buttons all converge on it. Changing the
dataset, model/role filters, x scale, or the Explorer conversation selection
RESETS the store — bin indices are positions in the current binning, and
keeping them across a re-bin would silently condition on different token
ranges.

One RENDER callback draws the three histograms and the inspector sections
from (filters, selections) — the inspector and the charts can never disagree.
"""
from __future__ import annotations

import logging

from dash import ALL, Input, Output, State, callback_context, html, no_update
from dash.exceptions import PreventUpdate

from modules import records
from modules.correlations_data import (add_bin_range, add_selection, bin_runs,
                                       bins_subset, delete_selection,
                                       initial_store, selection_color,
                                       set_applied, set_live, toggle_bin)
from modules.explorer_data import apply_selection
from modules.figures import (empty_figure, multi_histogram_figure,
                             selection_to_bins)
from modules.records import DIMENSIONS
from modules.theme import MONO
from modules.turns_callbacks import cached_dataset_options
from modules.turns_data import (bin_counts, dimension_values, edge_label,
                                filter_records, make_bins, value_stats)

logger = logging.getLogger(__name__)

_N_BINS = 60
# short dimension names for inspector headers ("Context length" etc.)
_SHORT = {d: DIMENSIONS[d]["label"].split("(")[0].strip() for d in DIMENSIONS}


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
        Output("at-corr-selections-store", "data"),
        Output("at-corr-hint", "children"),
        [Output(f"at-corr-graph-{dim}", "clickData") for dim in DIMENSIONS],
        [Input(f"at-corr-graph-{dim}", "clickData") for dim in DIMENSIONS],
        [Input(f"at-corr-graph-{dim}", "selectedData") for dim in DIMENSIONS],
        Input("at-corr-newsel-btn", "n_clicks"),
        Input("at-corr-apply-btn", "n_clicks"),
        Input("at-corr-clear-btn", "n_clicks"),
        Input({"type": "at-corr-insp-bin", "sid": ALL, "bin": ALL}, "n_clicks"),
        Input({"type": "at-corr-insp-live", "sid": ALL}, "n_clicks"),
        Input({"type": "at-corr-insp-del", "sid": ALL}, "n_clicks"),
        Input("at-corr-filter-store", "data"),
        Input("at-explorer-selection-store", "data"),
        State("at-corr-selections-store", "data"),
        prevent_initial_call=True,
    )
    def mutate_selections(*args):
        """Single owner of the selections-store (self-loop clickData outputs
        reset the graphs so re-clicking the same bin fires again)."""
        reset = [None] * len(DIMENSIONS)
        try:
            store = args[-1] or initial_store()
            ctx = callback_context
            trig = ctx.triggered_id
            tval = ctx.triggered[0]["value"] if ctx.triggered else None

            if trig in ("at-corr-filter-store", "at-explorer-selection-store"):
                # re-bin -> old bin indices would mean different token ranges
                return initial_store(), "", *reset
            if trig == "at-corr-clear-btn":
                if not tval:
                    raise PreventUpdate
                return initial_store(), "", *reset
            if trig == "at-corr-newsel-btn":
                if not tval:
                    raise PreventUpdate
                return add_selection(store), "", *reset
            if trig == "at-corr-apply-btn":
                if not tval:
                    raise PreventUpdate
                hint = ("" if any(s["bins"] for s in store["selections"])
                        else "nothing to apply yet — click bins on a histogram first")
                return set_applied(store, True), hint, *reset

            if isinstance(trig, str) and trig.startswith("at-corr-graph-"):
                if not tval:  # clickData reset echo / cleared select box
                    raise PreventUpdate
                dim = trig.removeprefix("at-corr-graph-")
                prop = ctx.triggered[0]["prop_id"].rsplit(".", 1)[1]
                if prop == "clickData":
                    pts = tval.get("points") or []
                    if not pts:
                        raise PreventUpdate
                    new = toggle_bin(store, store["live"], dim,
                                     int(round(pts[0]["x"])))
                else:  # selectedData box
                    rng = tval.get("range")
                    if not rng or "x" not in rng:
                        raise PreventUpdate
                    bins = selection_to_bins(tuple(rng["x"]), _N_BINS)
                    if not bins:
                        raise PreventUpdate
                    new = add_bin_range(store, store["live"], dim,
                                        bins[0], bins[-1])
                return new, "", *reset

            if isinstance(trig, dict):
                if not tval:  # initial n_clicks of freshly rendered sections
                    raise PreventUpdate
                sid = trig["sid"]
                if trig["type"] == "at-corr-insp-bin":
                    sel = next(s for s in store["selections"] if s["sid"] == sid)
                    if sel["dim"] is None:
                        raise PreventUpdate
                    new = toggle_bin(store, sid, sel["dim"], trig["bin"])
                elif trig["type"] == "at-corr-insp-live":
                    new = set_live(store, sid)
                elif trig["type"] == "at-corr-insp-del":
                    new = delete_selection(store, sid)
                else:
                    raise PreventUpdate
                return new, "", *reset

            raise PreventUpdate
        except PreventUpdate:
            raise
        except ValueError as e:  # e.g. click on a non-controlling histogram
            return no_update, str(e), *([no_update] * len(DIMENSIONS))
        except Exception:
            logger.exception("corr selection mutation failed")
            raise PreventUpdate

    @app.callback(
        [Output(f"at-corr-graph-{dim}", "figure") for dim in DIMENSIONS],
        Output("at-corr-inspectors", "children"),
        Output("at-corr-gate-status", "children"),
        Input("at-corr-filter-store", "data"),
        Input("at-corr-selections-store", "data"),
        Input("at-explorer-selection-store", "data"),
        prevent_initial_call=True,
    )
    def render(filters, store, conv_selection):
        try:
            if not filters or not filters.get("slug"):
                figs = [empty_figure(DIMENSIONS[d]["label"], "pick a dataset",
                                     height=None) for d in DIMENSIONS]
                return (*figs,
                        html.Div("pick a dataset",
                                 style={"fontSize": "11px", "color": "#999"}),
                        "")
            pool = records.load_records(filters["slug"])
            pool, sel_gates = apply_selection(pool, conv_selection, filters["slug"])
            filtered, gates = filter_records(
                pool, {"models": filters["models"], "roles": filters["roles"]})
            gates = {**sel_gates, **gates}
            log_x = filters["xscale"] == "log"
            store = store or initial_store()

            values_by_dim = {d: dimension_values(filtered, d) for d in DIMENSIONS}
            edges_by_dim = {d: make_bins(v, _N_BINS, log_x)
                            for d, v in values_by_dim.items() if v}
            counts_by_dim = {d: bin_counts(values_by_dim[d], e, log_x)
                             for d, e in edges_by_dim.items()}

            # matched subset per selection (None while a selection is empty)
            sel_rows = []
            for i, s in enumerate(store["selections"]):
                subset = None
                if s["dim"] in edges_by_dim and s["bins"]:
                    subset, _ = bins_subset(filtered, s["dim"],
                                            edges_by_dim[s["dim"]],
                                            s["bins"], log_x)
                sel_rows.append({"sel": s, "name": f"S{i + 1}",
                                 "color": selection_color(s["sid"]),
                                 "subset": subset})

            applied = bool(store.get("applied"))
            figs = []
            for d in DIMENSIONS:
                pv = values_by_dim[d]
                if not pv:
                    gate_txt = " → ".join(f"{k}={v}" for k, v in gates.items())
                    figs.append(empty_figure(
                        DIMENSIONS[d]["label"],
                        f"0 rows after filters ({gate_txt})", height=None))
                    continue
                own = [(r["sel"]["bins"], r["color"], r["name"])
                       for r in sel_rows
                       if r["sel"]["dim"] == d and r["sel"]["bins"]]
                overlays = [(dimension_values(r["subset"], d), r["color"], r["name"])
                            for r in sel_rows
                            if applied and r["subset"] is not None
                            and r["sel"]["dim"] != d] if applied else []
                figs.append(multi_histogram_figure(
                    pv, DIMENSIONS[d]["label"] + (" — conditioned" if overlays else ""),
                    log_x, _N_BINS, own_marks=own, overlays=overlays,
                    gates=gates))

            inspectors = [
                _inspector_section(r, store, edges_by_dim, counts_by_dim,
                                   len(filtered))
                for r in sel_rows
            ]
            gate_txt = "rows through gates:\n" + " → ".join(
                f"{k}={v:,}" for k, v in gates.items())
            gate_txt += ("\nconditioning: applied"
                         if applied else "\nconditioning: off — press Apply range")
            return *figs, inspectors, gate_txt
        except Exception:
            logger.exception("correlations render failed")
            raise PreventUpdate


def _inspector_section(row: dict, store: dict, edges_by_dim: dict,
                       counts_by_dim: dict, n_pool: int) -> html.Div:
    """One left-panel section per selection: clickable header (makes it
    live), delete button, a clickable per-bin strip of its controlling
    histogram, and per-bin/aggregate detail for the selected bins."""
    s, color, name = row["sel"], row["color"], row["name"]
    live = s["sid"] == store["live"]
    dim = s["dim"]

    header_children = [
        html.Span(style={"width": "10px", "height": "10px", "borderRadius": "5px",
                         "background": color, "flex": "0 0 auto"}),
        html.Span(f"{name} — {_SHORT[dim]}" if dim else f"{name} — empty"),
    ]
    if live:
        header_children.append(html.Span(
            "live", style={"fontSize": "10px", "color": "white",
                           "background": "#666", "borderRadius": "3px",
                           "padding": "0 4px"}))
    header = html.Div(
        header_children,
        id={"type": "at-corr-insp-live", "sid": s["sid"]}, n_clicks=0,
        title="Click to make this the live selection — histogram clicks and "
              "box-selects edit the live one.",
        style={"display": "flex", "alignItems": "center", "gap": "6px",
               "cursor": "pointer", "flex": "1 1 auto", "fontSize": "12px",
               "fontWeight": "700" if live else "400", "minWidth": "0"})
    del_btn = html.Button(
        "×", id={"type": "at-corr-insp-del", "sid": s["sid"]}, n_clicks=0,
        title="Delete this selection.",
        style={"fontSize": "11px", "padding": "0 6px", "lineHeight": "16px",
               "flex": "0 0 auto"})
    kids: list = [html.Div([header, del_btn],
                           style={"display": "flex", "alignItems": "center",
                                  "gap": "6px"})]

    if dim in counts_by_dim:
        counts, edges = counts_by_dim[dim], edges_by_dim[dim]
        max_c = max(counts) or 1
        bset = set(s["bins"])
        kids.append(html.Div(  # clickable mini-strip of ALL bins
            [html.Div(
                id={"type": "at-corr-insp-bin", "sid": s["sid"], "bin": b},
                n_clicks=0,
                title=(f"[{edge_label(edges[b])}, {edge_label(edges[b + 1])}) — "
                       f"{counts[b]:,} reqs — click to toggle"),
                style={"flex": "1 1 0", "minWidth": "0",
                       "height": f"{max(8, round(38 * counts[b] / max_c))}px",
                       "background": color if b in bset else "#c9c9c9",
                       "cursor": "pointer", "borderRight": "1px solid #fff"})
             for b in range(len(counts))],
            title="The controlling histogram's bins — click to add/remove.",
            style={"display": "flex", "alignItems": "flex-end", "height": "42px",
                   "marginTop": "5px", "background": "#f4f4f4",
                   "borderRadius": "3px", "overflow": "hidden"}))
        if s["bins"]:
            mono11 = {"fontFamily": MONO, "fontSize": "11px", "color": "#555"}
            subset = row["subset"] or []
            pct = 100 * len(subset) / n_pool if n_pool else 0.0
            kids.append(html.Div(
                f"{len(subset):,} matched ({pct:.1f}% of pool)",
                style={**mono11, "color": "#222", "marginTop": "4px"}))
            runs = bin_runs(s["bins"])
            run_lines = [
                f"[{edge_label(edges[a])}, {edge_label(edges[b + 1])}): "
                f"{sum(counts[a:b + 1]):,} reqs"
                for a, b in runs[:6]]
            if len(runs) > 6:
                run_lines.append(f"… +{len(runs) - 6} more ranges")
            kids += [html.Div(t, style=mono11) for t in run_lines]
            if subset:  # conditional stats of the OTHER dimensions
                for od in DIMENSIONS:
                    if od == dim:
                        continue
                    st = value_stats(dimension_values(subset, od))
                    kids.append(html.Div(
                        f"{_SHORT[od]}: med {edge_label(st['median'])}  "
                        f"mean {edge_label(st['mean'])}  "
                        f"p90 {edge_label(st['p90'])}",
                        style=mono11))
    else:
        kids.append(html.Div(
            "empty — click bins on a histogram (that chart becomes this "
            "selection's controlling histogram)",
            style={"fontSize": "11px", "color": "#999", "marginTop": "4px"}))

    return html.Div(kids, style={
        "border": f"1px solid {'#888' if live else '#ddd'}",
        "borderLeft": f"4px solid {color}", "borderRadius": "4px",
        "padding": "6px 8px", "marginTop": "8px", "background": "white"})
