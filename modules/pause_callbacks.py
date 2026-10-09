"""Pause Analytics callbacks: pauses of the shared conversation selection,
decomposed into causes and drawn on one shared duration axis."""
from __future__ import annotations

import logging

import plotly.graph_objects as go
from dash import Input, Output, State, html
from dash.exceptions import PreventUpdate

from modules import controls, records, theme
from modules.binning import edge_label
from modules.figures import empty_figure
from modules.pause_data import (UNKNOWN, all_pauses, cause_totals,
                                duration_bins, isl_by_bin, stack_by_cause)
from modules.pause_layout import ISL_MEASURES
from modules.theme import F_SMALL, PALETTE, fmt_count

logger = logging.getLogger(__name__)

_N_BINS = 40
_GRAY = "#9a9a9a"


def cause_color(cause: str, order: list[str]) -> str:
    """Stable per-cause colour; 'unknown' is always the neutral grey so the
    remainder never looks like a measured category."""
    if cause == UNKNOWN:
        return _GRAY
    return PALETTE[order.index(cause) % len(PALETTE)] if cause in order \
        else PALETTE[0]


def _pauses_for(slugs: list[str], selection: dict | None,
                min_s: float) -> list[dict]:
    """Pauses of the working set. An EMPTY conversation selection means
    every conversation - the explorer writes conv_ids=[] whenever nothing is
    ticked, and reading that as 'select nothing' emptied the whole tab."""
    pool = records.load_pool(slugs)
    ids = (selection or {}).get("conv_ids") or []
    sel = set(ids) if ids else None
    ps = all_pauses(pool, sel, records.load_pool_activities(slugs))
    return [p for p in ps if p["duration_s"] >= min_s]


def register_pause_callbacks(app) -> None:
    @app.callback(
        Output("at-pause-dataset-echo", "children"),
        Input("at-explorer-filter-store", "data"),
    )
    def echo_datasets(filters):
        slugs = (filters or {}).get("slugs") or []
        return "\n".join(slugs) if slugs else "none loaded (see Explorer)"

    @app.callback(
        Output("at-pause-causes-cl", "options"),
        Output("at-pause-causes-cl", "value"),
        Input("at-explorer-filter-store", "data"),
        Input("at-explorer-selection-store", "data"),
        Input("at-pause-min-input", "value"),
        State("at-pause-causes-cl", "value"),
        prevent_initial_call=True,
    )
    def cause_options(filters, selection, min_s, current):
        """Causes present in THIS dataset, biggest first. Everything is
        ticked by default; a cause that disappears from the data drops out
        of the selection rather than lingering as a phantom."""
        try:
            slugs = (filters or {}).get("slugs") or []
            if not slugs:
                return [], []
            ps = _pauses_for(slugs, selection, float(min_s or 0))
            if not ps:
                return [], []
            totals = cause_totals(ps)
            opts = [{"label": f"  {c}  ({v / 3600:,.1f} h)", "value": c}
                    for c, v in totals.items()]
            keep = [c for c in (current or []) if c in totals]
            return opts, keep or list(totals)
        except Exception:
            logger.exception("pause cause options failed")
            raise PreventUpdate

    @app.callback(
        Output("at-pause-stack-graph", "figure"),
        Output("at-pause-isl-graph", "figure"),
        Output("at-pause-summary-panel", "children"),
        Output("at-pause-gate-status", "children"),
        Input("at-explorer-filter-store", "data"),
        Input("at-explorer-selection-store", "data"),
        Input("at-pause-min-input", "value"),
        Input("at-pause-scale-radio", "value"),
        Input("at-pause-weight-radio", "value"),
        Input("at-pause-causes-cl", "value"),
        Input("at-pause-isl-radio", "value"),
        prevent_initial_call=True,
    )
    def render(filters, selection, min_s, scale, weight, causes, isl_field):
        try:
            slugs = (filters or {}).get("slugs") or []
            if not slugs:
                fig = empty_figure("Pause analytics",
                                   "load a dataset on Explorer", height=None)
                return fig, fig, html.Div("load a dataset", style={
                    "color": "#777", "fontSize": F_SMALL}), ""
            min_s = float(min_s or 0)
            log_x = scale != "linear"
            ps = _pauses_for(slugs, selection, min_s)
            if not ps:
                fig = empty_figure(
                    "Pause analytics",
                    f"no pause of {min_s:g}s or longer in this selection",
                    height=None)
                return fig, fig, html.Div("no pauses", style={
                    "color": "#777", "fontSize": F_SMALL}), \
                    f"pauses >= {min_s:g}s: 0"

            totals = cause_totals(ps)
            order = [c for c in totals if c != UNKNOWN]
            causes = [c for c in (causes or list(totals)) if c in totals] \
                or list(totals)
            edges = duration_bins(ps, _N_BINS, log_x)
            stack = stack_by_cause(ps, edges, log_x, causes, weight)
            isl = isl_by_bin(ps, edges, log_x, isl_field)

            n = len(edges) - 1
            idx = list(range(n))
            ticks = _ticks(edges)
            scale_div = 3600.0 if weight == "time" else 1.0
            y_title = "pause time (hours)" if weight == "time" else "pauses"

            f1 = go.Figure()
            for c in causes:
                f1.add_trace(go.Bar(
                    x=idx, y=[v / scale_div for v in stack[c]], name=c,
                    marker_color=cause_color(c, order), marker_line_width=0,
                    hovertemplate=(f"{c}<br>%{{y:,.2f}} {y_title}"
                                   "<extra></extra>")))
            f1.update_layout(**theme.base_layout(**theme.top_band_layout(
                f"Pauses by cause — {len(ps):,} pauses, "
                f"{sum(p['duration_s'] for p in ps) / 3600:,.1f} h total")))
            f1.update_layout(barmode="stack", bargap=0.05)
            _apply_ticks(f1, ticks, n, "pause duration (s)")
            f1.update_yaxes(title_text=y_title, title_font_size=11)

            f2 = go.Figure()
            hv = [i for i, m in enumerate(isl["median"]) if m is not None]
            f2.add_trace(go.Scatter(
                x=[i for i in hv], y=[isl["p90"][i] for i in hv], mode="lines",
                line=dict(width=0), hoverinfo="skip", showlegend=False))
            f2.add_trace(go.Scatter(
                x=[i for i in hv], y=[isl["p10"][i] for i in hv], mode="lines",
                line=dict(width=0), fill="tonexty",
                fillcolor="rgba(31,119,180,0.18)", name="p10–p90",
                hoverinfo="skip"))
            f2.add_trace(go.Scatter(
                x=[i for i in hv], y=[isl["median"][i] for i in hv],
                mode="lines+markers", line=dict(color="#1f77b4", width=2.2),
                marker=dict(size=5), name="median",
                customdata=[isl["n"][i] for i in hv],
                hovertemplate="median %{y:,.0f} tokens<br>%{customdata:,.0f} "
                              "pauses in this bin<extra></extra>"))
            f2.update_layout(**theme.base_layout(**theme.top_band_layout(
                ISL_MEASURES[isl_field] + " vs pause duration")))
            _apply_ticks(f2, ticks, n, "pause duration (s)")
            f2.update_yaxes(title_text="tokens", type="log", title_font_size=11)

            total_s = sum(p["duration_s"] for p in ps)
            panel = _summary(ps, totals, total_s, order, min_s)
            gate = (f"pauses >= {min_s:g}s: {len(ps):,}\n"
                    f"total pause time: {total_s / 3600:,.1f} h\n"
                    f"causes known: {100 * (1 - totals.get(UNKNOWN, 0) / total_s):.1f}%")
            return f1, f2, panel, gate
        except PreventUpdate:
            raise
        except Exception:
            logger.exception("pause render failed")
            raise PreventUpdate


def _ticks(edges: list[float]) -> list[tuple[float, str]]:
    n = len(edges) - 1
    step = max(1, n // 8)
    return [(v - 0.5, edge_label(edges[min(v, n)]))
            for v in range(0, n + 1, step)]


def _apply_ticks(fig: go.Figure, ticks, n: int, title: str) -> None:
    fig.update_xaxes(tickvals=[t[0] for t in ticks],
                     ticktext=[t[1] for t in ticks],
                     range=[-0.5, n - 0.5], title_text=title,
                     title_font_size=11)


def _summary(pauses: list[dict], totals: dict, total_s: float,
             order: list[str], min_s: float) -> list:
    longest = max(pauses, key=lambda p: p["duration_s"])
    rows = [(c, f"{v / 3600:,.2f} h  ({100 * v / total_s:.1f}%)")
            for c, v in totals.items()]
    durs = sorted(p["duration_s"] for p in pauses)
    return [
        controls.card("Pauses in this selection", [
            ("Pauses", f"{len(pauses):,}"),
            ("Total dead air", f"{total_s / 3600:,.1f} h"),
            ("Median", f"{durs[len(durs) // 2]:,.1f} s"),
            ("p90", f"{durs[int(.9 * len(durs))]:,.0f} s"),
            ("Longest", f"{longest['duration_s'] / 3600:,.2f} h"),
            ("… ended by", f"{fmt_count(longest['next_in_tokens'])} tok ctx"),
        ], info_text=f"Gaps of at least {min_s:g}s on each conversation's "
                     "main-agent lane, between the end of its last activity "
                     "and the start of the next request."),
        controls.card("Cause breakdown", rows,
                      info_text="An exact partition of the pause time: every "
                                "second is attributed to at most one cause, "
                                "and 'unknown' is the remainder the data "
                                "cannot explain — never a guess."),
    ]
