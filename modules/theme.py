"""Shared visual constants and formatting helpers.

Pure module: imports nothing from Dash. Plotly styling dicts live here so every
figure in the app hovers/looks the same.
"""

# --- Layout constants (imitating ix_explorer) ---
F_SMALL = "13px"
F_BASE = "15px"
LEFT_W = "280px"
MONO = "Consolas, 'Courier New', monospace"

# Hash-derived palette: stable colors across renders (index = hash % len).
PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3",
]

ROLE_COLORS = {"main": "#1f77b4", "subagent": "#ff7f0e"}

HOVERLABEL = dict(
    bgcolor="rgba(255,255,255,0.75)",
    bordercolor="#222",
    font=dict(family=MONO, size=12, color="#111"),
    align="left",
)

GRAPH_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToAdd": ["select2d", "lasso2d"],
    "scrollZoom": True,
    "responsive": True,  # figures follow their container / window size
}


def top_band_layout(title_text: str, title_size: int = 13) -> dict:
    """Title top-LEFT and legend top-RIGHT in one band above the plot, with
    enough top margin that neither clips nor overlaps the other."""
    return dict(
        title=dict(text=title_text, font=dict(size=title_size),
                   x=0.01, xanchor="left", y=0.985, yanchor="top"),
        margin=dict(l=55, r=15, t=58, b=42),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.0,
                    xanchor="right", x=1, font=dict(size=10)),
        autosize=True,
    )


def base_layout(**overrides) -> dict:
    """Common plotly layout: closest hover, tight margins, no built-in legend,
    visible row/column gridlines (the default template's white-on-white grid
    disappears against our near-white plot background)."""
    layout = dict(
        hovermode="closest",
        hoverlabel=HOVERLABEL,
        margin=dict(l=55, r=15, t=42, b=45),
        showlegend=False,
        plot_bgcolor="#fafafa",
        paper_bgcolor="white",
        font=dict(size=12),
        xaxis=dict(showgrid=True, gridcolor="#dcdcdc", gridwidth=1, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#dcdcdc", gridwidth=1, zeroline=False),
    )
    layout.update(overrides)
    return layout


def color_for(key: str) -> str:
    """Stable color for a string key (model name, series id)."""
    return PALETTE[hash(key) % len(PALETTE)]


# --- Number formatting ---

def fmt_count(n: float) -> str:
    """1234567 -> '1.23M'. Token/count formatting for labels and cards."""
    n = float(n)
    for div, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(n) >= div:
            return f"{n / div:.3g}{suffix}"
    return f"{n:.4g}"


def fmt_bytes(n_bytes: float) -> str:
    n = float(n_bytes)
    for div, suffix in ((1e15, "PB"), (1e12, "TB"), (1e9, "GB"), (1e6, "MB"), (1e3, "KB")):
        if abs(n) >= div:
            return f"{n / div:.3g} {suffix}"
    return f"{n:.4g} B"


def fmt_flops(n_flops: float) -> str:
    n = float(n_flops)
    for div, suffix in ((1e18, "EFLOPs"), (1e15, "PFLOPs"), (1e12, "TFLOPs"), (1e9, "GFLOPs")):
        if abs(n) >= div:
            return f"{n / div:.3g} {suffix}"
    return f"{n:.4g} FLOPs"


def fmt_seconds(s: float) -> str:
    s = float(s)
    if s >= 3600:
        return f"{s / 3600:.2f} GPU-h"
    if s >= 60:
        return f"{s / 60:.2f} GPU-min"
    return f"{s:.2f} GPU-s"
