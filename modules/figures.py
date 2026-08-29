"""Figure builders shared by the Turns and Correlations tabs.

Histograms are drawn as uniform-width bars over BIN INDEX (categorical), with
edge labels on the ticks — plotly bars on a true log axis misbehave, and index
bars make box-select return clean bin indices for the correlations bin picks.
"""
from __future__ import annotations

import math

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


def _apply_index_ticks(fig: go.Figure, edges: list[float], log_x: bool) -> None:
    """Edge labels on the index axis (every ~n/8th bin boundary)."""
    n = len(edges) - 1
    tick_step = max(1, n // 8)
    tickvals = list(range(0, n + 1, tick_step))
    fig.update_xaxes(
        tickvals=[v - 0.5 for v in tickvals],
        ticktext=[edge_label(edges[min(v, n)]) for v in tickvals],
        title_text=("tokens (log bins; 0→1)" if log_x else "tokens (linear bins)"),
        title_font_size=11,
    )


def _stats_annotation(fig: go.Figure, values: list[float]) -> None:
    stats = value_stats(values)
    fig.add_annotation(
        text=(f"n={stats['count']:,}  med={edge_label(stats['median'])}  "
              f"mean={edge_label(stats['mean'])}  p90={edge_label(stats['p90'])}  "
              f"max={edge_label(stats['max'])}"),
        showarrow=False, xref="paper", yref="paper", x=0.99, y=1.13,
        font=dict(family=theme.MONO, size=11, color="#444"), align="right",
    )


def histogram_figure(
    values: list[float],
    title: str,
    log_x: bool,
    n_bins: int,
    color: str = "#1f77b4",
    gates: dict | None = None,
) -> go.Figure:
    """One distribution as index-bars + stats annotation (Turns tab)."""
    if not values:
        gate_txt = " → ".join(f"{k}={v}" for k, v in (gates or {}).items())
        return empty_figure(title, f"0 rows after filters ({gate_txt or 'empty pool'})")

    edges = make_bins(values, n_bins, log_x)
    counts = bin_counts(values, edges, log_x)
    n = len(counts)
    hover = [
        f"[{edge_label(edges[i])}, {edge_label(edges[i + 1])})<br>count={counts[i]}"
        for i in range(n)
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(range(n)), y=counts, marker_color=color, marker_line_width=0,
        hovertext=hover, hoverinfo="text", name="",
    ))
    _apply_index_ticks(fig, edges, log_x)
    fig.update_yaxes(title_text="requests", title_font_size=11)
    fig.update_layout(**theme.base_layout(
        title=dict(text=title, font=dict(size=13)),
        height=260,
        bargap=0.05,
    ))
    _stats_annotation(fig, values)
    return fig


def multi_histogram_figure(
    pool_values: list[float],
    title: str,
    log_x: bool,
    n_bins: int,
    own_marks: list[tuple[list[int], str, str]],
    overlays: list[tuple[list[float], str, str]],
    gates: dict | None = None,
) -> go.Figure:
    """Correlations histogram: the FULL pool as a gray step silhouette (bin
    edges always come from the full pool — the chart never re-ranges), plus

      own_marks: [(bins, color, name)] — selections CONTROLLING this
        dimension; their picked bins are tinted at full pool height.
      overlays: [(values, color, name)] — matched-request values of applied
        selections controlling OTHER dimensions; drawn as colored bars
        stacked on each other (explicit base under barmode=overlay).

    A transparent full-height bar per bin is the topmost trace: it carries
    the combined hover text and makes ANY click inside a bin's column report
    that bin (clickData x = bin index).
    """
    if not pool_values:
        gate_txt = " → ".join(f"{k}={v}" for k, v in (gates or {}).items())
        return empty_figure(title, f"0 rows after filters ({gate_txt or 'empty pool'})",
                            height=None)

    edges = make_bins(pool_values, n_bins, log_x)
    pool_counts = bin_counts(pool_values, edges, log_x)
    n = len(pool_counts)
    idx = list(range(n))
    max_c = max(pool_counts) or 1

    overlay_counts = [(bin_counts(values, edges, log_x), color, name)
                      for values, color, name in overlays]

    hover = []
    for i in idx:
        lines = [f"[{edge_label(edges[i])}, {edge_label(edges[i + 1])}) — "
                 f"{pool_counts[i]:,} of all rows"]
        lines += [f"{name}: {cts[i]:,} matched"
                  for cts, _color, name in overlay_counts if cts[i]]
        lines.append("click to select / deselect this bin")
        hover.append("<br>".join(lines))

    fig = go.Figure()
    fig.add_trace(go.Scatter(  # full-pool silhouette (the grayed background)
        x=idx, y=pool_counts, mode="lines",
        line=dict(color="#999", width=1.2, shape="hvh"),
        fill="tozeroy", fillcolor="rgba(150,150,150,0.20)",
        hoverinfo="skip", name="all rows",
    ))
    for bins, color, name in own_marks:
        xs = [b for b in bins if 0 <= b < n]
        if not xs:
            continue
        fig.add_trace(go.Bar(
            x=xs, y=[pool_counts[b] for b in xs],
            marker_color=color, opacity=0.45, marker_line_width=0,
            hoverinfo="skip", name=f"{name} bins",
        ))
    base = [0.0] * n
    for cts, color, name in overlay_counts:
        fig.add_trace(go.Bar(
            x=idx, y=cts, base=list(base),
            marker_color=color, marker_line_width=0, opacity=0.9,
            hoverinfo="skip", name=name,
        ))
        base = [b + c for b, c in zip(base, cts)]
    fig.add_trace(go.Bar(  # transparent click/hover target, topmost
        x=idx, y=[max_c] * n, marker_color="rgba(0,0,0,0)",
        marker_line_width=0, hovertext=hover, hoverinfo="text", name="",
    ))

    _apply_index_ticks(fig, edges, log_x)
    fig.update_yaxes(title_text="requests", title_font_size=11)
    fig.update_layout(**theme.base_layout(
        title=dict(text=title, font=dict(size=13)),
        autosize=True,
        barmode="overlay",
        bargap=0.05,
        dragmode="select",
        selectdirection="h",
    ))
    _stats_annotation(fig, pool_values)
    return fig


def selection_to_bins(selected_range_x: tuple[float, float], n_bins: int) -> list[int]:
    """Map a box-select x-range on the INDEX axis to the bin indices whose
    column [i-0.5, i+0.5) the box overlaps (empty when the box lies entirely
    off the axis)."""
    x0, x1 = sorted(selected_range_x)
    lo_i = max(0, math.ceil(x0 - 0.5 + 1e-9))
    hi_i = min(n_bins - 1, math.floor(x1 + 0.5 - 1e-9))
    if hi_i < lo_i:
        return []
    return list(range(lo_i, hi_i + 1))
