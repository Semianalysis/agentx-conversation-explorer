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
from modules.arch import ARCHITECTURES, request_compute, resolve_assumptions
from modules.deepdive_data import (aggregate_selection, grouped_series,
                                   zoom_member_uids)
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
    )
    def coalesce_axes(x, y1, y2, y3, groupopts, xscale):
        groupopts = groupopts or []
        return {"x": x, "y1": y1, "y2": y2, "y3": y3,
                "grouped": "grouped" in groupopts,
                "envelope": "envelope" in groupopts,
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
            if zoom and zoom.get("chart") == slot:
                uid = pt.get("customdata")
                if not uid:  # grouped/envelope lines carry no request uid
                    raise PreventUpdate
                return no_update, {"uid": uid}, *reset
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
            zoom = zoom if zoom and zoom.get("chart") in _SLOTS else None
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
                    envelope=axes.get("envelope", False),
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
            if zoom and (point or {}).get("uid"):
                p = next((r for r in agg["per_request"]
                          if r["uid"] == point["uid"]), None)
                if p is not None:
                    panel = _point_inspector(p, arch, cfg) + panel
            return (*figs, panel)
        except PreventUpdate:
            raise
        except Exception:
            logger.exception("deep-dive render failed")
            raise PreventUpdate


def _measure_figure(per_req: list[dict], x_key: str, y_key: str, n_convs: int,
                    grouped: bool = False, envelope: bool = False,
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
    if grouped and x_key != "conv_number" and n_convs > 1:
        g = grouped_series(per_req, x_key, y_key)
        if envelope:
            for name, ys in (("min of any conv", g["lo"]),
                             ("max of any conv", g["hi"])):
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


def _point_inspector(p: dict, arch: dict, cfg: dict) -> list:
    """Right-column cards for one clicked request: its sizes and timing, the
    implied per-phase work, and the serving assumptions it is priced under."""
    c = request_compute(arch, p["cached_tokens"], p["uncached_tokens"],
                        p["out_tokens"], cfg)
    dur_s = p["end_s"] - p["start_s"]
    moe = arch["n_experts"] > 0
    return [
        _card(f"Point — conv #{p['ord']}, turn {p['seq']} ({p['role']})", [
            ("Model", p["model"]),
            ("Start / duration", f"{p['start_s']:,.0f}s / {dur_s:,.1f}s",
             "Conversation-relative start, and the request's own wall-clock."),
            ("Busy at start", f"{p['busy_s']:,.0f}s",
             "Active seconds elapsed in this conversation when the request "
             "began (idle gaps excluded)."),
            ("Context (total ISL)", fmt_count(p["in_tokens"]),
             "Total input sequence length — every token the request attends "
             "over."),
            ("… cached context", fmt_count(p["cached_tokens"]),
             "Served from the prompt cache — no prefill compute."),
            ("… new input (midfill)", fmt_count(p["uncached_tokens"]),
             "Uncached tokens actually prefilled on top of the cached "
             "context."),
            ("OSL (decode)", fmt_count(p["out_tokens"]),
             "Output sequence length — tokens generated."),
            ("KV cache", f"{fmt_bytes(p['kv_bytes'])} "
                         f"({fmt_count(p['in_tokens'])} tok)",
             "Context KV footprint under the KV precision assumption."),
            ("Prefill FLOPs", fmt_flops(c["prefill_flops"])),
            ("Decode FLOPs", fmt_flops(c["decode_flops"])),
            ("Turn FLOPs", fmt_flops(p["flops"])),
        ], info_text="One request, as clicked on the magnified chart. Sizes "
                     "and timing come from the trace; FLOPs/KV are implied "
                     "under the assumptions below. Double-click a chart to "
                     "close the magnifier and this inspector."),
        _card("Serving assumptions at this point", [
            ("Architecture", arch["label"]),
            ("TP / PP / DP",
             f"{cfg['tp']} / {cfg.get('pp', 1)} / {cfg.get('dp', 1)}",
             "Tensor / pipeline / data parallelism from the Explorer "
             "assumption bar."),
            ("EP",
             (f"all-to-all over {arch['n_experts']} experts, "
              f"top-{arch['topk']} (no EP degree modeled)") if moe
             else "dense architecture — none",
             "Expert-parallel traffic is modeled as full dispatch+combine "
             "all-to-all; an explicit EP degree is not a knob yet."),
            ("Weights / KV precision",
             f"{cfg['dtype_weights']} / {cfg['dtype_kv']}"),
            ("MFU / MBU",
             f"{cfg['mfu'] * 100:.0f}% / {cfg['mbu'] * 100:.0f}%"),
            ("Disaggregation", "none assumed",
             "Prefill, midfill (prefill on cached context), and decode are "
             "priced as colocated on one pool — disaggregated serving is "
             "not modeled, and the traces don't record deployment topology."),
        ], info_text="What this point's implied numbers are priced under — "
                     "all global assumptions from the Explorer bar, applied "
                     "identically to every request."),
    ]


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
