"""Deep-dive tab callbacks: three same-x charts over picked measures,
aggregating the shared conversation selection (all conversations when nothing
is checked). The conversation list itself — data and checkbox sync — is owned
by explorer_callbacks (same object, two viewports).
"""
from __future__ import annotations

import logging

import plotly.graph_objects as go
from dash import Input, Output, State, callback_context, html, no_update
from dash.exceptions import PreventUpdate

from modules import api_client, controls, records, theme
from modules.arch import ARCHITECTURES, resolve_assumptions
from modules.deepdive_data import (aggregate_selection, grouped_series,
                                   sweep_points, zoom_member_uids)
from modules.explorer_data import build_conversation_table
from modules.figures import empty_figure
from modules.measures import (ALL_MEASURES, X_MEASURES, Y_MEASURES, axis_units,
                              measure_series, shared_axis_range, zoom_window)
from modules.theme import (F_SMALL, MONO, ROLE_COLORS, color_for, fmt_bytes,
                           fmt_count, fmt_flops)

logger = logging.getLogger(__name__)

_MAX_PER_CONV_LINES = 12  # above this, cumulative measures fall back to markers
_SLOTS = ("y1", "y2", "y3")
_ZOOM_FACTOR = 5.0
_GRAY = "rgba(160,160,160,0.25)"


def register_deepdive_callbacks(app) -> None:
    @app.callback(
        Output("at-deep-axes-store", "data"),
        Input("at-deep-x-radio", "value"),
        Input("at-deep-y1-radio", "value"),
        Input("at-deep-y2-radio", "value"),
        Input("at-deep-y3-radio", "value"),
        Input("at-deep-groupopts-cl", "value"),
        Input("at-deep-xscale-radio", "value"),
        Input("at-deep-envelope-radio", "value"),
    )
    def coalesce_axes(x, y1, y2, y3, groupopts, xscale, envelope):
        groupopts = groupopts or []
        return {"x": x, "y1": y1, "y2": y2, "y3": y3,
                "grouped": "grouped" in groupopts,
                "envelope": envelope or "p1090",
                "xscale": xscale or "auto"}

    @app.callback(
        Output("at-deep-zoom-store", "data"),
        Output("at-deep-point-store", "data"),
        [Output(f"at-deep-{s}-graph", "clickData") for s in _SLOTS],
        [Input(f"at-deep-{s}-graph", "clickData") for s in _SLOTS],
        [Input(f"at-deep-{s}-graph", "relayoutData") for s in _SLOTS],
        Input("at-deep-axes-store", "data"),
        Input("at-explorer-filter-store", "data"),
        Input("at-explorer-selection-store", "data"),
        Input("at-explorer-config-store", "data"),
        State("at-deep-zoom-store", "data"),
        prevent_initial_call=True,
    )
    def zoom_and_point(*args):
        """Magnifier state: a click on an unmagnified chart zooms it 5×
        around the point; a click on a point of the MAGNIFIED chart opens
        the point inspector; a plotly double-click (relayout autorange
        event) cancels both. Any re-bin trigger resets (clickData self-loop
        resets let the same point be clicked again)."""
        reset = [None] * len(_SLOTS)
        try:
            zoom = args[-1]
            ctx = callback_context
            trig = ctx.triggered_id
            tval = ctx.triggered[0]["value"] if ctx.triggered else None

            if trig in ("at-deep-axes-store", "at-explorer-filter-store",
                        "at-explorer-selection-store",
                        "at-explorer-config-store"):
                return None, None, *reset
            if not (isinstance(trig, str) and trig.startswith("at-deep-")):
                raise PreventUpdate
            slot = trig.removeprefix("at-deep-").removesuffix("-graph")
            prop = ctx.triggered[0]["prop_id"].rsplit(".", 1)[1]
            if prop == "relayoutData":
                if tval and ("xaxis.autorange" in tval
                             or "yaxis.autorange" in tval):
                    return None, None, *reset  # double-click = cancel
                raise PreventUpdate
            if not tval:  # clickData reset echo
                raise PreventUpdate
            pts = tval.get("points") or []
            if not pts:
                raise PreventUpdate
            pt = pts[0]
            axes = args[6] or {}
            if axes.get("grouped"):
                # grouped mode has no magnifier; a click inspects the x
                # interval (the mean line only has points at occupied bins,
                # so idle cumulative-time stretches are unclickable)
                return None, {"interval_x": pt["x"]}, *reset
            if zoom and zoom.get("chart") == slot:
                # inspect the clicked x interval - same inspector as grouped
                # mode (single-request cards were useless for cumulative
                # measures; the user inspects the turn interval)
                return no_update, {"interval_x": pt["x"]}, *reset
            return ({"chart": slot, "cx": pt["x"], "cy": pt["y"]},
                    None, *reset)
        except PreventUpdate:
            raise
        except Exception:
            logger.exception("deep-dive zoom/point failed")
            raise PreventUpdate

    @app.callback(
        Output("at-deep-y1-graph", "figure"),
        Output("at-deep-y2-graph", "figure"),
        Output("at-deep-y3-graph", "figure"),
        Output("at-deep-summary-panel", "children"),
        Input("at-explorer-selection-store", "data"),
        Input("at-explorer-config-store", "data"),
        Input("at-explorer-filter-store", "data"),
        Input("at-deep-axes-store", "data"),
        Input("at-deep-zoom-store", "data"),
        Input("at-deep-point-store", "data"),
        prevent_initial_call=True,
    )
    def render(selection, config, filters, axes, zoom, point):
        try:
            slug = (filters or {}).get("slug")
            if not slug:
                fig = empty_figure("Conversation deep-dive",
                                   "pick a dataset on Explorer", height=None)
                return fig, fig, fig, html.Div(
                    "pick a dataset", style={"color": "#777", "fontSize": F_SMALL})
            axes = axes or {}
            for name in ("x", "y1", "y2", "y3"):
                if axes.get(name) not in ALL_MEASURES:
                    raise PreventUpdate  # radios not initialized yet
            arch_key, _gpu_key, cfg = resolve_assumptions(config)
            arch = ARCHITECTURES[arch_key]

            pool = records.load_records(slug)
            index_ids = [it["conv_id"] for it in api_client.fetch_conversation_index(slug)]
            rows, _ = build_conversation_table(pool, index_ids, None)
            ordinal = {r["id"]: r["ordinal"] for r in rows}
            sel_ids = list((selection or {}).get("conv_ids") or []) \
                if (selection or {}).get("slug") == slug else []
            all_selected = not sel_ids
            conv_ids = sel_ids or list(ordinal)

            agg = aggregate_selection(pool, conv_ids, arch, cfg, ordinal)

            xscale = axes.get("xscale", "auto")
            x_scale_eff = (X_MEASURES[axes["x"]]["scale"] if xscale == "auto"
                           else xscale)
            # One locked x window for all three charts (grouped-mode bin mids
            # sit at most half a bin inside the data extremes — the 2% pad
            # covers that, so the same range fits every chart mode).
            x_range = shared_axis_range(
                measure_series(agg["per_request"], axes["x"]), x_scale_eff)

            # magnifier: window on the zoomed chart, member set for graying
            # (no magnifier in grouped mode - clicks inspect intervals there)
            zoom = zoom if zoom and zoom.get("chart") in _SLOTS else None
            if axes.get("grouped"):
                zoom = None
            window_x = window_y = None
            member_uids: set | None = None
            if zoom:
                z_y_key = axes[zoom["chart"]]
                y_range = shared_axis_range(
                    measure_series(agg["per_request"], z_y_key), "log")
                window_x = zoom_window(axis_units(max(zoom["cx"], 1e-12),
                                                  x_scale_eff),
                                       x_range, _ZOOM_FACTOR)
                window_y = zoom_window(axis_units(max(zoom["cy"], 1e-12),
                                                  "log"),
                                       y_range, _ZOOM_FACTOR)
                member_uids = zoom_member_uids(
                    agg["per_request"], axes["x"], z_y_key,
                    window_x, window_y, x_scale_eff)

            figs = []
            for slot in _SLOTS:
                is_zoomed = bool(zoom) and zoom["chart"] == slot
                fig = _measure_figure(
                    agg["per_request"], axes["x"], axes[slot],
                    agg["n_convs"],
                    grouped=axes.get("grouped", False),
                    envelope=axes.get("envelope", "p1090"),
                    x_scale=x_scale_eff,
                    member_uids=(None if (is_zoomed or member_uids is None)
                                 else member_uids),
                    magnified=is_zoomed)
                if is_zoomed:
                    fig.update_xaxes(range=window_x, autorange=False)
                    fig.update_yaxes(range=window_y, autorange=False)
                else:
                    fig.update_xaxes(range=x_range, autorange=False)
                figs.append(fig)

            panel = _summary_panel(arch, cfg, agg, all_selected)
            if (point or {}).get("interval_x") is not None and (
                    zoom or axes.get("grouped")):
                panel = _interval_inspector(
                    agg["per_request"], axes, point["interval_x"]) + panel
            return (*figs, panel)
        except PreventUpdate:
            raise
        except Exception:
            logger.exception("deep-dive render failed")
            raise PreventUpdate


    @app.callback(
        Output("at-deep-sweep-dl", "data"),
        Input("at-deep-sweep-btn", "n_clicks"),
        State("at-explorer-selection-store", "data"),
        State("at-explorer-config-store", "data"),
        State("at-explorer-filter-store", "data"),
        State("at-deep-axes-store", "data"),
        prevent_initial_call=True,
    )
    def export_sweep(n, selection, config, filters, axes):
        """Download simulator sweep points sampled from the current grouped
        mean (10 evenly spaced x positions, averages of the contributing
        requests; see deepdive_data.sweep_points)."""
        try:
            if not n:
                raise PreventUpdate
            slug = (filters or {}).get("slug")
            axes = axes or {}
            if not slug or axes.get("x") not in X_MEASURES:
                raise PreventUpdate
            import json as _json

            from dash import dcc as _dcc
            arch_key, _g, cfg = resolve_assumptions(config)
            pool = records.load_records(slug)
            index_ids = [it["conv_id"]
                         for it in api_client.fetch_conversation_index(slug)]
            ordinal = {cid: i + 1 for i, cid in enumerate(index_ids)}
            sel_ids = list((selection or {}).get("conv_ids") or [])                 if (selection or {}).get("slug") == slug else []
            agg = aggregate_selection(pool, sel_ids or index_ids,
                                      ARCHITECTURES[arch_key], cfg, ordinal)
            sweep = sweep_points(agg["per_request"], axes["x"])
            sweep.update(dataset=slug,
                         selection=f"{len(sel_ids) or 'all'} conversations")
            return _dcc.send_string(
                _json.dumps(sweep, indent=1),
                f"sweep-{slug}-{axes['x']}.json")
        except PreventUpdate:
            raise
        except Exception:
            logger.exception("sweep export failed")
            raise PreventUpdate


def _measure_figure(per_req: list[dict], x_key: str, y_key: str, n_convs: int,
                    grouped: bool = False, envelope: str = "p1090",
                    x_scale: str | None = None,
                    member_uids: set | None = None,
                    magnified: bool = False) -> go.Figure:
    """x_scale overrides the x measure's natural axis scale ('linear'/'log');
    None keeps the registry's scale. member_uids = the magnifier window's
    member set: on a NON-magnified chart, requests outside it draw gray
    (marker mode only — line modes aren't grayed per point). magnified marks
    the title."""
    xm, ym = dict(X_MEASURES[x_key]), Y_MEASURES[y_key]
    if x_scale:
        if x_scale not in ("linear", "log"):
            raise ValueError(f"unknown x_scale override {x_scale!r}")
        xm["scale"] = x_scale
    fig = go.Figure()
    title = ym["label"] + (" — magnified 5×" if magnified else "")
    # n_convs == 1 still draws the grouped line (the mean of one
    # conversation IS that conversation, binned) — a >1 guard here made the
    # mean toggle silently dead for single-conversation selections
    if grouped and x_key != "conv_number":
        g = grouped_series(per_req, x_key, y_key)
        band = {"minmax": (("min of any conv", g["lo"]),
                           ("max of any conv", g["hi"])),
                "p1090": (("p10 of convs", g["p10"]),
                          ("p90 of convs", g["p90"]))}.get(envelope, ())
        if band:
            for name, ys in band:
                fig.add_trace(go.Scattergl(
                    x=g["xs"], y=ys, mode="lines",
                    line=dict(color="#888", width=1.4, dash="dot"),
                    name=name,
                    hovertemplate=(name + " — x=%{x:,.6g}  y=%{y:,.6g}"
                                   "<extra></extra>"),
                ))
        fig.add_trace(go.Scattergl(
            x=g["xs"], y=g["mean"], mode="lines",
            line=dict(color="#1f77b4", width=2.4),
            name="mean of alive convs",
            customdata=g["n_alive"],
            hovertemplate=("mean — x=%{x:,.6g}  y=%{y:,.6g}<br>"
                           "%{customdata} convs still alive<extra></extra>"),
        ))
        title += f" — grouped mean of {n_convs} convs (drop-out, no padding)"
        return _finish_measure_figure(fig, title, xm, ym)
    draw_lines = (ym.get("cumulative") and x_key != "conv_number"
                  and n_convs <= _MAX_PER_CONV_LINES)
    if draw_lines:
        by_conv: dict[str, list[dict]] = {}
        for p in per_req:
            by_conv.setdefault(p["cid"], []).append(p)
        for cid, rows in sorted(by_conv.items(), key=lambda kv: kv[1][0]["ord"]):
            rows = sorted(rows, key=xm["getter"])
            fig.add_trace(go.Scattergl(
                x=[xm["getter"](p) for p in rows],
                y=[ym["getter"](p) for p in rows],
                mode="lines", line=dict(color=color_for(cid), width=2),
                name=f"#{rows[0]['ord']}",
                customdata=[p["uid"] for p in rows],
                hovertemplate=(f"conv #{rows[0]['ord']} — "
                               "x=%{x:,.6g}  y=%{y:,.6g}<extra></extra>"),
            ))
    else:
        def _marker_trace(rows, color, name, opacity):
            fig.add_trace(go.Scattergl(
                x=[xm["getter"](p) for p in rows],
                y=[ym["getter"](p) for p in rows],
                mode="markers",
                marker=dict(color=color, size=4, opacity=opacity),
                name=name,
                customdata=[p["uid"] for p in rows],
                hovertext=[(f"conv #{p['ord']} — {p['model']} ({p['role']})<br>"
                            f"turn {p['seq']}  t={p['start_s']:.0f}s<br>"
                            f"x={xm['getter'](p):,.6g}  y={ym['getter'](p):,.6g}")
                           for p in rows],
                hoverinfo="text",
            ))

        for role, color in ROLE_COLORS.items():
            rows = [p for p in per_req if p["role"] == role]
            if not rows:
                continue
            if member_uids is None:
                _marker_trace(rows, color, role, 0.5)
                continue
            inside = [p for p in rows if p["uid"] in member_uids]
            outside = [p for p in rows if p["uid"] not in member_uids]
            if outside:
                _marker_trace(outside, _GRAY, f"{role} (outside magnifier)",
                              0.35)
            if inside:
                _marker_trace(inside, color, role, 0.6)
    return _finish_measure_figure(fig, title, xm, ym)


def _finish_measure_figure(fig: go.Figure, title: str, xm: dict, ym: dict) -> go.Figure:
    """Shared axes/title/legend finishing: title top-left, legend top-right,
    no fixed height — the figure fills its flex container (config.responsive)."""
    fig.update_layout(**theme.base_layout(**theme.top_band_layout(title)))
    fig.update_xaxes(title_text=xm["label"], title_font_size=11,
                     type=("log" if xm["scale"] == "log" else "linear"))
    fig.update_yaxes(title_font_size=11,
                     type=("log" if ym["scale"] == "log" else "linear"))
    return fig


def _card(title: str, rows: list[tuple], info_text: str | None = None) -> html.Div:
    """Summary card. rows are (key, value) or (key, value, hover-help)."""
    title_children: list = [title]
    if info_text:
        title_children.append(controls.info(info_text))
    row_divs = []
    for row in rows:
        k, v = row[0], row[1]
        tip = row[2] if len(row) > 2 else None
        row_divs.append(html.Div(
            style={"display": "flex", "justifyContent": "space-between",
                   "fontSize": "12px", "gap": "10px"},
            children=[html.Span([k, controls.info(tip)] if tip else k,
                                style={"color": "#666"}),
                      html.Span(v, style={"fontFamily": MONO})]))
    return html.Div(
        style={"border": "1px solid #ddd", "borderRadius": "6px", "padding": "10px",
               "marginBottom": "10px", "background": "white"},
        children=[html.Div(title_children,
                           style={"fontWeight": "600", "fontSize": F_SMALL,
                                  "marginBottom": "6px", "display": "flex",
                                  "alignItems": "center"})] + row_divs,
    )


def _interval_inspector(per_request: list[dict], axes: dict,
                        interval_x: float) -> list:
    """Right-column cards for one clicked x interval (grouped mean line OR a
    magnified point cloud): which rows fall in it, mean turn FLOPs, and a
    mini histogram of each chart's y measure over exactly those rows.
    x=turn # -> the interval IS that turn; time x -> the geometric bin the
    click landed in. Cumulative measures are histogrammed as their PER-TURN
    counterparts (running totals are meaningless inside one interval)."""
    import math

    from dash import dcc

    from modules.binning import bin_counts, edge_label, make_bins

    x_key = axes["x"]
    xg = ALL_MEASURES[x_key]["getter"]
    if x_key == "conv_number":
        return []
    if x_key == "turn_number":
        turn = round(interval_x)
        rows = [p for p in per_request if p["seq"] == turn]
        label = f"turn {turn}"
    else:
        xs_all = [max(xg(p), 1.0) for p in per_request]
        lo_x, hi_x = 1.0, max(max(xs_all), 1.0)
        if hi_x == lo_x:
            hi_x = lo_x * 2
        log_ratio = math.log(hi_x / lo_x) / 60
        b = max(0, min(int(math.log(max(interval_x, 1.0) / lo_x) / log_ratio),
                       59))
        b_lo, b_hi = lo_x * math.exp(log_ratio * b), lo_x * math.exp(
            log_ratio * (b + 1))
        rows = [p for i, p in enumerate(per_request)
                if b_lo <= xs_all[i] < b_hi]
        label = f"[{edge_label(b_lo)}, {edge_label(b_hi)}) s"
    if not rows:
        return [_card("Interval inspector",
                      [("no requests", f"in {label}")],
                      info_text="The clicked interval holds no requests.")]

    n = len(rows)
    mean_flops = sum(p["flops"] for p in rows) / n
    kids = [html.Div(
        [f"Interval — {ALL_MEASURES[x_key]['label']} {label}, "
         f"{n:,} requests, "
         f"{len({p['cid'] for p in rows})} conversations",
         controls.info("Distributions of each chart's y measure over exactly "
                       "the requests in the clicked interval (cumulative "
                       "measures are shown as their PER-TURN counterparts — "
                       "running totals are meaningless inside one interval). "
                       "Double-click a chart to dismiss.")],
        style={"fontWeight": "600", "fontSize": F_SMALL, "display": "flex",
               "alignItems": "center", "marginBottom": "4px"}),
        html.Div(f"mean turn FLOPs: {fmt_flops(mean_flops)}",
                 title="Average implied FLOPs (prefill + decode) per request "
                       "in this interval, under the current assumptions.",
                 style={"fontFamily": MONO, "fontSize": "11px",
                        "color": "#555", "marginBottom": "2px"})]
    _PER_TURN = {"cumulative_flops": "current_flops",
                 "cumulative_output_tokens": "new_output_tokens"}
    for slot in _SLOTS:
        ym = Y_MEASURES[_PER_TURN.get(axes[slot], axes[slot])]
        values = [ym["getter"](p) for p in rows]
        edges = make_bins(values, 15, log_x=True)
        counts = bin_counts(values, edges, log_x=True)
        fig = go.Figure(go.Bar(
            x=list(range(len(counts))), y=counts, marker_color="#1f77b4",
            marker_line_width=0,
            hovertext=[f"[{edge_label(edges[i])}, {edge_label(edges[i + 1])})"
                       f" - {c}" for i, c in enumerate(counts)],
            hoverinfo="text"))
        tick = max(1, len(counts) // 4)
        fig.update_layout(**theme.base_layout(
            title=dict(text=ym["label"], font=dict(size=11)),
            height=130, margin=dict(l=30, r=8, t=24, b=18), bargap=0.05))
        fig.update_xaxes(tickvals=[v - 0.5 for v in range(0, len(counts) + 1,
                                                          tick)],
                         ticktext=[edge_label(edges[min(v, len(counts))])
                                   for v in range(0, len(counts) + 1, tick)],
                         tickfont_size=9)
        fig.update_yaxes(tickfont_size=9)
        kids.append(dcc.Graph(figure=fig,
                              config={"displayModeBar": False},
                              style={"height": "130px"}))
    return [html.Div(kids, style={
        "border": "1px solid #ddd", "borderRadius": "6px", "padding": "10px",
        "marginBottom": "10px", "background": "white"})]


def _summary_panel(arch, cfg, agg, all_selected: bool) -> list:
    totals = agg["totals"]
    wall_s = agg["wall_s_sum"]
    n_sub = sum(1 for p in agg["per_request"] if p["role"] == "subagent")
    scope = (f"All {agg['n_convs']} conversations (nothing checked)"
             if all_selected else f"Selection — {agg['n_convs']} conversations")
    return [
        _card(scope, [
            ("Requests", f"{totals['n_requests']:,}"),
            ("… subagent", f"{n_sub:,}",
             "Requests made by nested agents spawned via tool calls."),
            ("Wall-clock (sum)", f"{wall_s / 3600:.2f} h",
             "Sum of each conversation's span (first request start to last "
             "request end) — includes idle time, and conversations overlap "
             "in real time, so this is workload volume, not elapsed time."),
            ("Input tokens", fmt_count(totals["in_tokens"])),
            ("… cached", fmt_count(totals["cached_tokens"]),
             "Input tokens served from the prompt cache — no prefill compute."),
            ("… uncached", fmt_count(totals["uncached_tokens"]),
             "New input tokens actually prefilled."),
            ("Output tokens", fmt_count(totals["out_tokens"])),
        ], info_text="Token counts come straight from the traces; everything "
                     "below is IMPLIED from them under the assumption bar's "
                     "settings on the Explorer tab."),
        _card(f"Implied compute — {arch['label']}", [
            ("Prefill FLOPs", fmt_flops(totals["prefill_flops"]),
             "Processing new input: linear layers over uncached tokens plus "
             "attention against each request's context."),
            ("Decode FLOPs", fmt_flops(totals["decode_flops"]),
             "Generating output tokens, one forward pass per token."),
            ("Total FLOPs", fmt_flops(totals["prefill_flops"] + totals["decode_flops"])),
        ], info_text="FLOPs this architecture WOULD spend serving these "
                     "traces — computed from the token counts, not measured "
                     "on real hardware."),
        _card(f"Implied memory movement (HBM, kv {cfg['dtype_kv']})", [
            ("Prefill", fmt_bytes(totals["prefill_hbm_bytes"])),
            ("Decode", fmt_bytes(totals["decode_hbm_bytes"]),
             "Decode re-reads the weights and the growing KV cache for every "
             "generated token — usually the bandwidth-bound phase."),
            ("Total", fmt_bytes(totals["prefill_hbm_bytes"] + totals["decode_hbm_bytes"])),
            ("Peak KV footprint (one conv)", fmt_bytes(totals["peak_kv_bytes"]),
             "Largest single-request context KV under these assumptions — "
             "what one replica must hold in HBM at that moment."),
        ], info_text="Bytes moved through GPU memory (HBM): weight reads and "
                     "KV-cache reads/writes implied by the token counts."),
        _card(f"Implied network (TP={cfg['tp']}, PP={cfg.get('pp', 1)}, "
              f"DP={cfg.get('dp', 1)})", [
            ("TP all-reduce", fmt_bytes(totals["net_tp_bytes"]),
             "Tensor parallelism synchronizes every layer's partial results "
             "across the TP group — twice per layer per token."),
            ("EP all-to-all", fmt_bytes(totals["net_ep_bytes"]),
             "MoE expert dispatch and combine traffic (zero for dense "
             "architectures)."),
            ("PP stage traffic", fmt_bytes(totals["net_pp_bytes"]),
             "Activations crossing each pipeline-stage boundary (zero when "
             "PP=1)."),
        ], info_text="Interconnect traffic implied by the parallelism "
                     "assumptions from the Explorer assumption bar."),
    ]
