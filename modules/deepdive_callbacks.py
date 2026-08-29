"""Deep-dive tab callbacks: three same-x charts over picked measures,
aggregating the shared conversation selection (all conversations when nothing
is checked). The conversation list itself — data and checkbox sync — is owned
by explorer_callbacks (same object, two viewports).
"""
from __future__ import annotations

import logging

import plotly.graph_objects as go
from dash import Input, Output, html
from dash.exceptions import PreventUpdate

from modules import api_client, controls, records, theme
from modules.arch import ARCHITECTURES, GPUS, implied_gpu_seconds, resolve_assumptions
from modules.deepdive_data import aggregate_selection, grouped_series
from modules.explorer_data import build_conversation_table
from modules.figures import empty_figure
from modules.measures import (ALL_MEASURES, X_MEASURES, Y_MEASURES,
                              measure_series, shared_axis_range)
from modules.theme import (F_SMALL, MONO, ROLE_COLORS, color_for, fmt_bytes,
                           fmt_count, fmt_flops, fmt_seconds)

logger = logging.getLogger(__name__)

_MAX_PER_CONV_LINES = 12  # above this, cumulative measures fall back to markers


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
        Output("at-deep-y1-graph", "figure"),
        Output("at-deep-y2-graph", "figure"),
        Output("at-deep-y3-graph", "figure"),
        Output("at-deep-summary-panel", "children"),
        Input("at-explorer-selection-store", "data"),
        Input("at-explorer-config-store", "data"),
        Input("at-explorer-filter-store", "data"),
        Input("at-deep-axes-store", "data"),
        prevent_initial_call=True,
    )
    def render(selection, config, filters, axes):
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
            arch_key, gpu_key, cfg = resolve_assumptions(config)
            arch = ARCHITECTURES[arch_key]
            gpu = GPUS[gpu_key]

            pool = records.load_records(slug)
            index_ids = [it["conv_id"] for it in api_client.fetch_conversation_index(slug)]
            rows, _ = build_conversation_table(pool, index_ids, None)
            ordinal = {r["id"]: r["ordinal"] for r in rows}
            sel_ids = list((selection or {}).get("conv_ids") or []) \
                if (selection or {}).get("slug") == slug else []
            all_selected = not sel_ids
            conv_ids = sel_ids or list(ordinal)

            agg = aggregate_selection(pool, conv_ids, arch, cfg, ordinal)
            gpu_time = implied_gpu_seconds(agg["totals"], gpu, cfg)

            xscale = axes.get("xscale", "auto")
            x_scale_eff = (X_MEASURES[axes["x"]]["scale"] if xscale == "auto"
                           else xscale)
            figs = [
                _measure_figure(agg["per_request"], axes["x"], axes[y_name],
                                agg["n_convs"],
                                grouped=axes.get("grouped", False),
                                envelope=axes.get("envelope", False),
                                x_scale=x_scale_eff)
                for y_name in ("y1", "y2", "y3")
            ]
            # One locked x window for all three charts (grouped-mode bin mids
            # sit at most half a bin inside the data extremes — the 2% pad
            # covers that, so the same range fits every chart mode).
            x_range = shared_axis_range(
                measure_series(agg["per_request"], axes["x"]), x_scale_eff)
            for fig in figs:
                fig.update_xaxes(range=x_range, autorange=False)
            return (*figs, _summary_panel(arch, gpu, cfg, agg, gpu_time,
                                          all_selected))
        except PreventUpdate:
            raise
        except Exception:
            logger.exception("deep-dive render failed")
            raise PreventUpdate


def _measure_figure(per_req: list[dict], x_key: str, y_key: str, n_convs: int,
                    grouped: bool = False, envelope: bool = False,
                    x_scale: str | None = None) -> go.Figure:
    """x_scale overrides the x measure's natural axis scale ('linear'/'log');
    None keeps the registry's scale."""
    xm, ym = dict(X_MEASURES[x_key]), Y_MEASURES[y_key]
    if x_scale:
        if x_scale not in ("linear", "log"):
            raise ValueError(f"unknown x_scale override {x_scale!r}")
        xm["scale"] = x_scale
    fig = go.Figure()
    title = ym["label"]
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
                hovertemplate=(f"conv #{rows[0]['ord']} — "
                               "x=%{x:,.6g}  y=%{y:,.6g}<extra></extra>"),
            ))
    else:
        for role, color in ROLE_COLORS.items():
            rows = [p for p in per_req if p["role"] == role]
            if not rows:
                continue
            fig.add_trace(go.Scattergl(
                x=[xm["getter"](p) for p in rows],
                y=[ym["getter"](p) for p in rows],
                mode="markers",
                marker=dict(color=color, size=4, opacity=0.5),
                name=role,
                hovertext=[(f"conv #{p['ord']} — {p['model']} ({p['role']})<br>"
                            f"turn {p['seq']}  t={p['start_s']:.0f}s<br>"
                            f"x={xm['getter'](p):,.6g}  y={ym['getter'](p):,.6g}")
                           for p in rows],
                hoverinfo="text",
            ))
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


def _summary_panel(arch, gpu, cfg, agg, gpu_time, all_selected: bool) -> list:
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
        _card(f"Single-GPU-equivalent time — {gpu['label']} "
              f"({cfg['dtype_weights']}, MFU {cfg['mfu'] * 100:.0f}%, "
              f"MBU {cfg['mbu'] * 100:.0f}%)", [
            ("Prefill", fmt_seconds(gpu_time["prefill_s"])),
            ("… bound by", gpu_time["prefill_bound"],
             "Whichever takes longer decides: compute (FLOPs ÷ peak×MFU) or "
             "memory (bytes ÷ bandwidth×MBU)."),
            ("Decode", fmt_seconds(gpu_time["decode_s"])),
            ("… bound by", gpu_time["decode_bound"],
             "Whichever takes longer decides: compute (FLOPs ÷ peak×MFU) or "
             "memory (bytes ÷ bandwidth×MBU)."),
            ("Total", fmt_seconds(gpu_time["total_s"])),
            ("Sustained GPUs (vs summed wall)",
             f"{gpu_time['total_s'] / wall_s:.3g}" if wall_s > 0 else "n/a",
             "GPU-seconds ÷ summed wall-clock: the average number of GPUs "
             "this workload keeps busy end-to-end, idle time included."),
        ], info_text="How long ONE such GPU would need for all the implied "
                     "work, at the assumed utilization — the basis for the "
                     "sustained-GPU counts in the conversation list."),
    ]
