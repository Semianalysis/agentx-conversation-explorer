"""Correlations tab callbacks.

One MUTATION callback owns the selections-store: bin clicks and box-selects
on the three charts (honored only on the shared selection chart — or any
chart while nothing is anchored yet), clicks on inspector strips / arming
headers / color-coded Clear buttons, and the Add-selection button all
converge on it. Changing the dataset, model/role filters, measures, x scale,
serving assumptions, or the Explorer conversation selection RESETS the store
— bin indices are positions in the current binning, and keeping them across
a re-bin would silently condition on different value ranges.

One RENDER callback draws the three charts, the armed-cursor classes on
their wrappers, and the inspector sections from (filters, axes, selections)
— the inspectors and the charts can never disagree.
"""
from __future__ import annotations

import logging

from dash import ALL, Input, Output, State, callback_context, html, no_update
from dash.exceptions import PreventUpdate

from modules import api_client, records
from modules.arch import ARCHITECTURES, resolve_assumptions
from modules.binning import (bin_counts, bin_weighted, edge_label,
                             filter_records, make_bins)
from modules.correlations_data import (ALL_MEASURES, CHART_SLOTS, DEFAULT_AXES,
                                       X_MEASURES, Y_MEASURES, add_selection,
                                       assign_bin, assign_bin_range, bin_runs,
                                       clear_selection, initial_store,
                                       measure_values, member_mask,
                                       selection_color, set_live)
from modules.deepdive_data import aggregate_selection
from modules.explorer_callbacks import cached_dataset_options
from modules.figures import (empty_figure, multi_histogram_figure,
                             selection_to_bins)
from modules.theme import MONO, fmt_count

logger = logging.getLogger(__name__)

_N_BINS = 60

# enriched-rows memo: bin clicks re-render constantly; re-enriching 393
# conversations each time would make every click lag
_ENRICH_CACHE: dict = {}


def _enriched_rows(slug: str, conv_selection: dict | None, config: dict | None,
                   ) -> list[dict]:
    """Per-request rows enriched with seq/start_s/busy_s/kv_bytes/flops for
    the Explorer-selected conversations (all when nothing selected), under
    the global serving assumptions. Memoized on (slug, selection, config)."""
    pool = records.load_records(slug)
    sel_ids = tuple(sorted((conv_selection or {}).get("conv_ids") or [])) \
        if (conv_selection or {}).get("slug") == slug else ()
    arch_key, _gpu_key, cfg = resolve_assumptions(config)
    key = (slug, sel_ids, arch_key, tuple(sorted(cfg.items())))
    if key in _ENRICH_CACHE:
        return _ENRICH_CACHE[key]
    index = api_client.fetch_conversation_index(slug)
    ordinal = {it["conv_id"]: i + 1 for i, it in enumerate(index)}
    conv_ids = list(sel_ids) or records.conversation_ids(pool)
    rows = aggregate_selection(pool, conv_ids, ARCHITECTURES[arch_key], cfg,
                               ordinal)["per_request"]
    _ENRICH_CACHE.clear()  # keep exactly the latest working set
    _ENRICH_CACHE[key] = rows
    return rows


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
        Output("at-corr-axes-store", "data"),
        Input("at-corr-x-radio", "value"),
        Input("at-corr-y1-radio", "value"),
        Input("at-corr-y2-radio", "value"),
        Input("at-corr-y3-radio", "value"),
    )
    def coalesce_axes(x, y1, y2, y3):
        return {"x": x, "y1": y1, "y2": y2, "y3": y3}

    @app.callback(
        Output("at-corr-selections-store", "data"),
        Output("at-corr-hint", "children"),
        [Output(f"at-corr-graph-{slot}", "clickData") for slot in CHART_SLOTS],
        [Input(f"at-corr-graph-{slot}", "clickData") for slot in CHART_SLOTS],
        [Input(f"at-corr-graph-{slot}", "selectedData") for slot in CHART_SLOTS],
        Input("at-corr-newsel-btn", "n_clicks"),
        Input({"type": "at-corr-insp-bin", "sid": ALL, "bin": ALL}, "n_clicks"),
        Input({"type": "at-corr-insp-live", "sid": ALL}, "n_clicks"),
        Input({"type": "at-corr-insp-clear", "sid": ALL}, "n_clicks"),
        Input("at-corr-filter-store", "data"),
        Input("at-corr-axes-store", "data"),
        Input("at-explorer-selection-store", "data"),
        Input("at-explorer-config-store", "data"),
        State("at-corr-selections-store", "data"),
        prevent_initial_call=True,
    )
    def mutate_selections(*args):
        """Single owner of the selections-store (self-loop clickData outputs
        reset the graphs so re-clicking the same bin fires again)."""
        reset = [None] * len(CHART_SLOTS)
        try:
            store = args[-1] or initial_store()
            ctx = callback_context
            trig = ctx.triggered_id
            tval = ctx.triggered[0]["value"] if ctx.triggered else None

            if trig in ("at-corr-filter-store", "at-corr-axes-store",
                        "at-explorer-selection-store",
                        "at-explorer-config-store"):
                # re-bin -> old bin indices would mean different value ranges
                return initial_store(), "", *reset
            if trig == "at-corr-newsel-btn":
                if not tval:
                    raise PreventUpdate
                return add_selection(store), "", *reset

            if isinstance(trig, str) and trig.startswith("at-corr-graph-"):
                if not tval:  # clickData reset echo / cleared select box
                    raise PreventUpdate
                slot = trig.removeprefix("at-corr-graph-")
                prop = ctx.triggered[0]["prop_id"].rsplit(".", 1)[1]
                try:
                    if prop == "clickData":
                        pts = tval.get("points") or []
                        if not pts:
                            raise PreventUpdate
                        new = assign_bin(store, store["live"], slot,
                                         int(round(pts[0]["x"])))
                    else:  # selectedData box
                        rng = tval.get("range")
                        if not rng or "x" not in rng:
                            raise PreventUpdate
                        bins = selection_to_bins(tuple(rng["x"]), _N_BINS)
                        if not bins:
                            raise PreventUpdate
                        new = assign_bin_range(store, store["live"], slot,
                                               bins[0], bins[-1])
                except ValueError:
                    # click on a non-selection chart: the default cursor
                    # already says this chart is inert — stay silent
                    raise PreventUpdate
                return new, "", *reset

            if isinstance(trig, dict):
                if not tval:  # initial n_clicks of freshly rendered sections
                    raise PreventUpdate
                sid = trig["sid"]
                if trig["type"] == "at-corr-insp-bin":
                    if store["chart"] is None:
                        raise PreventUpdate
                    # editing a strip also arms that inspector
                    new = set_live(assign_bin(store, sid, store["chart"],
                                              trig["bin"]), sid)
                elif trig["type"] == "at-corr-insp-live":
                    new = set_live(store, sid)
                elif trig["type"] == "at-corr-insp-clear":
                    new = clear_selection(store, sid)
                else:
                    raise PreventUpdate
                return new, "", *reset

            raise PreventUpdate
        except PreventUpdate:
            raise
        except ValueError as e:  # e.g. selection cap reached
            return no_update, str(e), *([no_update] * len(CHART_SLOTS))
        except Exception:
            logger.exception("corr selection mutation failed")
            raise PreventUpdate

    @app.callback(
        [Output(f"at-corr-graph-{slot}", "figure") for slot in CHART_SLOTS],
        [Output(f"at-corr-wrap-{slot}", "className") for slot in CHART_SLOTS],
        Output("at-corr-inspectors", "children"),
        Output("at-corr-selection-status", "children"),
        Output("at-corr-gate-status", "children"),
        Input("at-corr-filter-store", "data"),
        Input("at-corr-axes-store", "data"),
        Input("at-corr-selections-store", "data"),
        Input("at-explorer-selection-store", "data"),
        Input("at-explorer-config-store", "data"),
        prevent_initial_call=True,
    )
    def render(filters, axes, store, conv_selection, config):
        try:
            no_cursor = [""] * len(CHART_SLOTS)
            if not filters or not filters.get("slug"):
                figs = [empty_figure("Correlations", "pick a dataset",
                                     height=None) for _ in CHART_SLOTS]
                return (*figs, *no_cursor,
                        html.Div("pick a dataset",
                                 style={"fontSize": "11px", "color": "#999"}),
                        "", "")
            axes = axes or DEFAULT_AXES
            if (axes.get("x") not in X_MEASURES
                    or any(axes.get(s) not in Y_MEASURES for s in CHART_SLOTS)):
                raise PreventUpdate  # radios not initialized yet
            store = store or initial_store()
            log_x = filters["xscale"] == "log"
            token_mode = axes["x"] == "token_count"

            enriched = _enriched_rows(filters["slug"], conv_selection, config)
            filtered, gates = filter_records(
                enriched, {"models": filters["models"], "roles": filters["roles"]})
            if not filtered:
                gate_txt = " → ".join(f"{k}={v:,}" for k, v in gates.items())
                figs = [empty_figure(Y_MEASURES[axes[s]]["label"],
                                     f"0 rows after filters ({gate_txt})",
                                     height=None) for s in CHART_SLOTS]
                return (*figs, *no_cursor, [], "", gate_txt)

            # per-chart binning variable, edges, and pool heights
            y_vals = {s: measure_values(filtered, axes[s]) for s in CHART_SLOTS}
            if token_mode:
                bin_vals = dict(y_vals)  # each chart bins its own measure
                edges = {s: make_bins(bin_vals[s], _N_BINS, log_x)
                         for s in CHART_SLOTS}
                heights = {s: [float(c) for c in
                               bin_counts(bin_vals[s], edges[s], log_x)]
                           for s in CHART_SLOTS}
                y_titles = {s: "requests" for s in CHART_SLOTS}
                x_titles = {s: Y_MEASURES[axes[s]]["label"] for s in CHART_SLOTS}
            else:
                x_vals = measure_values(filtered, axes["x"])
                shared = make_bins(x_vals, _N_BINS, log_x)
                bin_vals = {s: x_vals for s in CHART_SLOTS}
                edges = {s: shared for s in CHART_SLOTS}
                heights = {s: bin_weighted(x_vals, shared, log_x, y_vals[s])
                           for s in CHART_SLOTS}
                y_titles = {s: f"Σ {Y_MEASURES[axes[s]]['label']}"
                            for s in CHART_SLOTS}
                x_titles = {s: X_MEASURES[axes["x"]]["label"]
                            for s in CHART_SLOTS}

            # member rows per non-empty selection (on the selection chart)
            sel_chart = store["chart"]
            sel_rows = []
            for i, s in enumerate(store["selections"]):
                members = None
                if sel_chart and s["bins"]:
                    mask = member_mask(bin_vals[sel_chart], edges[sel_chart],
                                       s["bins"], log_x)
                    members = [j for j, m in enumerate(mask) if m]
                sel_rows.append({"sel": s, "name": f"S{i + 1}",
                                 "color": selection_color(s["sid"]),
                                 "members": members})

            figs = []
            for c in CHART_SLOTS:
                own = [(r["sel"]["bins"], r["color"], r["name"])
                       for r in sel_rows
                       if c == sel_chart and r["sel"]["bins"]]
                overlays = []
                if c != sel_chart:
                    for r in sel_rows:
                        if r["members"] is None:
                            continue
                        if token_mode:
                            mvals = [bin_vals[c][j] for j in r["members"]]
                            hts = [float(v) for v in
                                   bin_counts(mvals, edges[c], log_x)]
                        else:
                            mx = [bin_vals[c][j] for j in r["members"]]
                            mw = [y_vals[c][j] for j in r["members"]]
                            hts = bin_weighted(mx, edges[c], log_x, mw)
                        overlays.append((hts, r["color"], r["name"]))
                title = Y_MEASURES[axes[c]]["label"] + (
                    " — conditioned" if overlays else "")
                figs.append(multi_histogram_figure(
                    edges[c], heights[c], title, x_titles[c], y_titles[c],
                    log_x, own_marks=own, overlays=overlays,
                    stat_values=y_vals[c]))

            # armed cursor: on the selection chart, or everywhere while
            # nothing is anchored yet
            armed_cls = f"corr-cursor-{store['live'] % 10}"
            classes = [armed_cls if (sel_chart is None or c == sel_chart)
                       else "" for c in CHART_SLOTS]

            inspectors = [
                _inspector_section(r, store, axes, edges, heights, y_vals,
                                   token_mode, len(filtered))
                for r in sel_rows
            ]
            status = (f"selections are in "
                      f"{Y_MEASURES[axes[sel_chart]]['label']}"
                      if sel_chart
                      else "you may choose to begin a selection in any chart")
            gate_txt = "rows through gates:\n" + " → ".join(
                f"{k}={v:,}" for k, v in gates.items())
            return *figs, *classes, inspectors, status, gate_txt
        except PreventUpdate:
            raise
        except Exception:
            logger.exception("correlations render failed")
            raise PreventUpdate


def _rgba(hex_color: str, alpha: float) -> str:
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def _inspector_section(row: dict, store: dict, axes: dict, edges: dict,
                       heights: dict, y_vals: dict, token_mode: bool,
                       n_pool: int) -> html.Div:
    """One left-panel section per selection: an arming header with the
    colored cursor-arrow (click to pick with this color), a color-coded
    Clear button, a clickable per-bin strip of the shared selection chart
    (own bins in this color, bins owned by OTHER inspectors in their owner's
    faded color), and per-bin / aggregate detail. Sections never disappear —
    an unused selection is just empty."""
    s, color, name = row["sel"], row["color"], row["name"]
    armed = s["sid"] == store["live"]
    sel_chart = store["chart"]

    n_bins_owned = len(s["bins"])
    header = html.Div(
        [html.Span("➤", style={"color": color, "fontSize": "15px",
                               "flex": "0 0 auto",
                               "opacity": 1.0 if armed else 0.35}),
         html.Span(f"{name} — {n_bins_owned} bins" if n_bins_owned
                   else f"{name} — empty"),
         *( [html.Span("picking", style={"fontSize": "10px", "color": "white",
                                         "background": color,
                                         "borderRadius": "3px",
                                         "padding": "0 4px"})] if armed else [] )],
        id={"type": "at-corr-insp-live", "sid": s["sid"]}, n_clicks=0,
        title="Click to arm this inspector: the mouse cursor takes its color "
              "over the selection chart (over every chart while nothing is "
              "selected yet) and your bin picks land here. Picking a bin "
              "another inspector owns transfers it; picking one this "
              "inspector owns releases it.",
        style={"display": "flex", "alignItems": "center", "gap": "6px",
               "cursor": "pointer", "flex": "1 1 auto", "fontSize": "12px",
               "fontWeight": "700" if armed else "400", "minWidth": "0"})
    clear_btn = html.Button(
        "Clear", id={"type": "at-corr-insp-clear", "sid": s["sid"]}, n_clicks=0,
        title="Empty this selection. The section stays for reuse.",
        style={"fontSize": "11px", "padding": "0 8px", "lineHeight": "18px",
               "flex": "0 0 auto", "borderRadius": "3px",
               "border": f"1px solid {color}", "background": "white",
               "color": color, "cursor": "pointer"})
    kids: list = [html.Div([header, clear_btn],
                           style={"display": "flex", "alignItems": "center",
                                  "gap": "6px"})]

    if sel_chart:
        e, h = edges[sel_chart], heights[sel_chart]
        max_h = max(h) or 1
        owner_by_bin = {}
        for other in store["selections"]:
            for b in other["bins"]:
                owner_by_bin[b] = other["sid"]
        own = set(s["bins"])
        strip_bins = []
        for b in range(len(h)):
            owner = owner_by_bin.get(b)
            if b in own:
                bg = color
            elif owner is not None:
                bg = _rgba(selection_color(owner), 0.35)
            else:
                bg = "#c9c9c9"
            strip_bins.append(html.Div(
                id={"type": "at-corr-insp-bin", "sid": s["sid"], "bin": b},
                n_clicks=0,
                title=(f"[{edge_label(e[b])}, {edge_label(e[b + 1])}) — "
                       f"{fmt_count(h[b])} — click to toggle for {name}"),
                style={"flex": "1 1 0", "minWidth": "0",
                       "height": f"{max(8, round(38 * h[b] / max_h))}px",
                       "background": bg, "cursor": "pointer",
                       "borderRight": "1px solid #fff"}))
        kids.append(html.Div(
            strip_bins,
            title="The selection chart's bins — click to add/remove for this "
                  "inspector (faded colors mark bins owned by other "
                  "inspectors; clicking transfers them here).",
            style={"display": "flex", "alignItems": "flex-end", "height": "42px",
                   "marginTop": "5px", "background": "#f4f4f4",
                   "borderRadius": "3px", "overflow": "hidden"}))
        if s["bins"]:
            mono11 = {"fontFamily": MONO, "fontSize": "11px", "color": "#555"}
            members = row["members"] or []
            pct = 100 * len(members) / n_pool if n_pool else 0.0
            kids.append(html.Div(
                f"{len(members):,} requests matched ({pct:.1f}% of pool)",
                style={**mono11, "color": "#222", "marginTop": "4px"}))
            runs = bin_runs(s["bins"])
            run_lines = [
                f"[{edge_label(e[a])}, {edge_label(e[b + 1])}): "
                f"{fmt_count(sum(h[a:b + 1]))}"
                for a, b in runs[:6]]
            if len(runs) > 6:
                run_lines.append(f"… +{len(runs) - 6} more ranges")
            kids += [html.Div(t, style=mono11) for t in run_lines]
            for c in CHART_SLOTS:  # contribution to each other chart
                if c == sel_chart:
                    continue
                total = sum(y_vals[c][j] for j in members)
                kids.append(html.Div(
                    f"Σ {Y_MEASURES[axes[c]]['label']}: {fmt_count(total)}",
                    style=mono11))
    else:
        kids.append(html.Div(
            "empty — click bins on any chart (the first pick chooses the "
            "selection chart)",
            style={"fontSize": "11px", "color": "#999", "marginTop": "4px"}))

    return html.Div(kids, style={
        "border": f"1px solid {'#888' if armed else '#ddd'}",
        "borderLeft": f"4px solid {color}", "borderRadius": "4px",
        "padding": "6px 8px", "marginTop": "8px", "background": "white"})
