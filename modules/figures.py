"""Figure builders shared by the Turns and Correlations tabs.

Histograms are drawn as uniform-width bars over BIN INDEX (categorical), with
edge labels on the ticks — plotly bars on a true log axis misbehave, and index
bars make box-select return clean bin indices for the correlations range pick.
"""
from __future__ import annotations

import plotly.graph_objects as go

from modules import theme
from modules.turns_data import bin_counts, edge_label, make_bins, value_stats


def empty_figure(title: str, reason: str, height: int | None = 260) -> go.Figure:
    """Diagnostic empty chart: names WHICH gate ate the rows, never blank.
    height=None -> autosize to the container (flex-filled charts)."""
    fig = go.Figure()
    size = {"height": height} if height else {"autosize": True}
    fig.update_layout(**theme.base_layout(title=title, **size))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.add_annotation(text=reason, showarrow=False, font=dict(size=13, color="#a33"),
                       xref="paper", yref="paper", x=0.5, y=0.5)
    return fig


def histogram_figure(
    values: list[float],
    title: str,
    log_x: bool,
    n_bins: int,
    color: str = "#1f77b4",
    gates: dict | None = None,
    highlight_range: tuple[float, float] | None = None,
    overlay_values: list[float] | None = None,
    overlay_scale: float = 1.0,
    edge_values: list[float] | None = None,
) -> go.Figure:
    """One distribution as index-bars + stats annotation.

    highlight_range: (lo, hi) token range to shade (condition band).
    overlay_values: faint outline of a reference distribution binned on the
    SAME edges, counts multiplied by overlay_scale (correlations: full pool
    rescaled to the conditional subset's total for shape comparison).
    edge_values: values to derive bin edges from when they must cover a
    superset of `values` (defaults to `values`).
    """
    if not values:
        gate_txt = " → ".join(f"{k}={v}" for k, v in (gates or {}).items())
        return empty_figure(title, f"0 rows after filters ({gate_txt or 'empty pool'})")

    edges = make_bins(edge_values if edge_values else values, n_bins, log_x)
    counts = bin_counts(values, edges, log_x)
    stats = value_stats(values)
    n = len(counts)
    idx = list(range(n))
    hover = [
        f"[{edge_label(edges[i])}, {edge_label(edges[i + 1])})<br>count={counts[i]}"
        for i in range(n)
    ]

    fig = go.Figure()
    if overlay_values:
        over_counts = [c * overlay_scale for c in bin_counts(overlay_values, edges, log_x)]
        fig.add_trace(go.Scatter(
            x=idx, y=over_counts, mode="lines",
            line=dict(color="#999", width=1.2, shape="hvh"),
            hoverinfo="skip", name="all rows (rescaled)",
        ))
    fig.add_trace(go.Bar(
        x=idx, y=counts, marker_color=color, marker_line_width=0,
        hovertext=hover, hoverinfo="text", name="",
    ))

    tick_step = max(1, n // 8)
    tickvals = list(range(0, n + 1, tick_step))
    fig.update_xaxes(
        tickvals=[v - 0.5 for v in tickvals],
        ticktext=[edge_label(edges[min(v, n)]) for v in tickvals],
        title_text=("tokens (log bins; 0→1)" if log_x else "tokens (linear bins)"),
        title_font_size=11,
    )
    fig.update_yaxes(title_text="requests", title_font_size=11)
    fig.update_layout(**theme.base_layout(
        title=dict(text=title, font=dict(size=13)),
        height=260,
        bargap=0.05,
        dragmode="select",
        selectdirection="h",
    ))
    fig.add_annotation(
        text=(f"n={stats['count']:,}  med={edge_label(stats['median'])}  "
              f"mean={edge_label(stats['mean'])}  p90={edge_label(stats['p90'])}  "
              f"max={edge_label(stats['max'])}"),
        showarrow=False, xref="paper", yref="paper", x=0.99, y=1.13,
        font=dict(family=theme.MONO, size=11, color="#444"), align="right",
    )

    if highlight_range is not None:
        lo_tok, hi_tok = highlight_range
        lo_i = _token_to_index(edges, lo_tok)
        hi_i = _token_to_index(edges, hi_tok)
        fig.add_vrect(x0=lo_i - 0.5, x1=hi_i + 0.5,
                      fillcolor="rgba(214,39,40,0.10)",
                      line=dict(color="rgba(214,39,40,0.6)", width=1))
    return fig


def _token_to_index(edges: list[float], tok: float) -> float:
    """Fractional bar-index position of a token value on the index axis."""
    n = len(edges) - 1
    if tok <= edges[0]:
        return -0.5
    if tok >= edges[-1]:
        return n - 0.5
    for i in range(n):
        if edges[i] <= tok < edges[i + 1]:
            frac = (tok - edges[i]) / (edges[i + 1] - edges[i])
            return i - 0.5 + frac
    return n - 0.5


def selection_to_token_range(
    selected_range_x: tuple[float, float],
    values: list[float],
    n_bins: int,
    log_x: bool,
) -> tuple[float, float]:
    """Map a box-select x-range on the INDEX axis back to token bounds using the
    same edges the figure was built with."""
    edges = make_bins(values, n_bins, log_x)
    n = len(edges) - 1
    lo_i = max(0, min(n - 1, int(round(selected_range_x[0] + 0.5))))
    hi_i = max(0, min(n - 1, int(round(selected_range_x[1] - 0.5))))
    if hi_i < lo_i:
        lo_i, hi_i = hi_i, lo_i
    return edges[lo_i], edges[hi_i + 1]
