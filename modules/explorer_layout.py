"""Explorer tab layout: growth-curve chart (most of the screen) + scrollable,
sortable, filterable conversation list.

Serving assumptions (GPU, quantizations, TP/PP/DP, MFU, MBU, architecture) are
compact dropdowns in a TOP BAR, all defaulting to 'any' (= the documented
default); their resolved values appear as columns in the list, and the list —
via column sorting (click; clicks stack for 2nd/3rd-order sort) and the native
per-column filter row (supports ranges: >100000, <=2) — is the main filtering
surface. The left column keeps only dataset + selection controls.

Cross-tab stores live HERE (all tabs stay mounted, see app.py):
  at-explorer-filter-store     {'slug'}
  at-explorer-selection-store  {'slug', 'conv_ids': [...]}
  at-explorer-config-store     serving assumptions (None = 'any')
  at-explorer-visible-store    conv_ids surviving the list's native filters
"""
from __future__ import annotations

from dash import dash_table, dcc, html

from modules import controls
from modules.arch import ARCHITECTURES, DEFAULT_ASSUMPTIONS, GPUS
from modules.controls import NONE
from modules.theme import F_SMALL, GRAPH_CONFIG, MONO

# d3-format thousands grouping as a PLAIN dict (not a Format object): the sort
# callback re-outputs columns to annotate sort badges, and plain dicts survive
# the JSON round-trip.
_GROUPED = {"specifier": ","}

# Shared by the Explorer and Deep-dive conversation lists (one source of truth).
TABLE_COLUMNS = [
    {"name": "#", "id": "ordinal", "type": "numeric"},
    {"name": "model", "id": "model", "type": "text"},
    {"name": "turns", "id": "n_turns", "type": "numeric", "format": _GROUPED},
    {"name": "initial ISL", "id": "isl0", "type": "numeric", "format": _GROUPED},
    {"name": "max ctx", "id": "max_ctx", "type": "numeric", "format": _GROUPED},
    {"name": "final ctx", "id": "final_ctx", "type": "numeric", "format": _GROUPED},
    {"name": "avg out", "id": "avg_out", "type": "numeric"},
    {"name": "max out", "id": "max_out", "type": "numeric", "format": _GROUPED},
    {"name": "hours", "id": "duration_h", "type": "numeric"},
    {"name": "GPU", "id": "gpu", "type": "text"},
    {"name": "Wq", "id": "wq", "type": "text"},
    {"name": "KVq", "id": "kvq", "type": "text"},
    {"name": "TP", "id": "tp", "type": "numeric"},
    {"name": "PP", "id": "pp", "type": "numeric"},
    {"name": "DP", "id": "dp", "type": "numeric"},
    {"name": "prefill GPUs", "id": "prefill_gpus", "type": "numeric"},
    {"name": "decode GPUs", "id": "decode_gpus", "type": "numeric"},
]

# Header hover-help for the shared conversation list (both viewports). Keyed
# by column id, so the sort-badge callback rewriting column NAMES can't
# orphan them.
COLUMN_TOOLTIPS = {
    "ordinal": "Conversation number — its rank in the dataset's token-sorted "
               "index (1 = most total input tokens). Stable identity for the "
               "conversation everywhere in the app.",
    "model": "Main-agent model of the conversation (subagent requests may use "
             "other models).",
    "n_turns": "Main-agent requests in the conversation. Subagent requests "
               "are not counted here.",
    "isl0": "Initial input sequence length — input tokens of the very first "
            "request (system prompt + first user message).",
    "max_ctx": "Largest single-request input (tokens) anywhere in the "
               "conversation. Often exceeds 'final ctx' because compaction "
               "shrinks the context mid-conversation.",
    "final_ctx": "Input tokens of the conversation's last main-agent request.",
    "avg_out": "Mean decoded (output) tokens per main-agent request. NOT "
               "(final ctx − initial ISL) ÷ turns — outputs are consumed by "
               "tools and compaction, not accumulated into context.",
    "max_out": "Largest single-request decoded output (tokens).",
    "duration_h": "Wall-clock hours from first request start to last request "
                  "end — includes idle time when nothing ran (typically most "
                  "of the span).",
    "gpu": "GPU type resolved from the assumption bar ('any' → default). Not "
           "recorded in the traces — a serving assumption.",
    "wq": "Weight precision assumed for the implied-compute math "
          "(bf16 = 2 bytes/param, fp8 = 1).",
    "kvq": "KV-cache precision assumed (bf16 = 2 bytes/entry, fp8 = 1).",
    "tp": "Tensor-parallel degree — how many GPUs split each layer's matmuls.",
    "pp": "Pipeline-parallel degree — how many sequential GPU stages the "
          "layers are split into.",
    "dp": "Data-parallel degree — independent model replicas serving "
          "different requests.",
    "prefill_gpus": "Sustained GPU count implied by the trace's prefill work: "
                    "implied prefill GPU-seconds ÷ the conversation's "
                    "wall-clock span. Empty when the span is 0.",
    "decode_gpus": "Sustained GPU count implied by the trace's decode work: "
                   "implied decode GPU-seconds ÷ wall-clock span. Empty when "
                   "the span is 0.",
}

# Kwargs shared by both DataTables so header help can never diverge.
TABLE_TOOLTIP_KWARGS = dict(tooltip_header=COLUMN_TOOLTIPS, tooltip_delay=400,
                            tooltip_duration=None)


def _any_dd(id_: str, options: list[dict], width: str = "150px") -> dcc.Dropdown:
    return dcc.Dropdown(
        id=id_, clearable=False, value=NONE,
        options=[{"label": "any", "value": NONE}] + options,
        style={"fontSize": "12px", "width": width})


def _bar_item(label: str, component, info: str | None = None) -> html.Div:
    label_children: list = [label]
    if info:
        label_children.append(controls.info(info))
    return html.Div(
        style={"display": "flex", "flexDirection": "column", "gap": "1px"},
        children=[html.Span(label_children,
                            style={"fontSize": "10px", "color": "#666",
                                   "fontWeight": "600"}),
                  component])


def _num_options(values: list[int]) -> list[dict]:
    return [{"label": str(v), "value": v} for v in values]


def layout() -> html.Div:
    side = controls.sidebar([
        controls.label("Dataset",
                       info_text="Pick which cached dataset to explore — every "
                                 "tab follows this choice. Datasets are "
                                 "imported and downloaded on the Overview tab."),
        controls.dropdown("at-explorer-dataset-dd", "pick a cached dataset"),
        controls.label("Chart x scale (turn count)",
                       info_text="Scale of the growth chart's x axis (main-"
                                 "agent turn count). Log compresses very long "
                                 "conversations so short ones stay readable."),
        dcc.RadioItems(
            id="at-explorer-xscale-radio",
            options=[{"label": " linear", "value": "linear"},
                     {"label": " log", "value": "log"}],
            value="linear", style={"fontSize": F_SMALL}),
        dcc.Checklist(
            id="at-explorer-onlysel-cl",
            options=[{"label": html.Span([
                " draw only selected conversations",
                controls.info("Hide unselected curves instead of dimming "
                              "them. With nothing selected, all curves draw.")],
                style={"display": "inline"}), "value": "only"}],
            value=[], style={"fontSize": F_SMALL, "marginTop": "6px"}),
        html.Button("Clear selection", id="at-explorer-clear-btn", n_clicks=0,
                    title="Empty the cross-tab selection — every tab falls "
                          "back to all conversations.",
                    style={"fontSize": F_SMALL, "marginTop": "12px"}),
        html.Div(id="at-explorer-selection-status",
                 style={"fontSize": "12px", "color": "#666", "margin": "8px 0",
                        "fontFamily": MONO, "whiteSpace": "pre-wrap"}),
        html.Div("Select conversations with the list checkboxes (shift-click toggles "
                 "a whole range), a cell drag, or by clicking curves. The Deep-dive "
                 "list is the same object — checks carry over both ways. Filter with "
                 "the column filter row (e.g. >100000 under 'final ctx'); click "
                 "column headers to sort — clicks stack for 2nd/3rd-order sorts.",
                 style={"fontSize": "11px", "color": "#999", "marginTop": "10px"}),
        html.Div("Serving assumptions live in the bar above the chart; 'any' means "
                 "the documented default. GPU counts in the list are the sustained "
                 "single-GPU-equivalents implied by each trace under those "
                 "assumptions.",
                 style={"fontSize": "11px", "color": "#999", "marginTop": "10px"}),
    ])

    assumption_bar = html.Div(
        style={"display": "flex", "gap": "10px", "alignItems": "flex-end",
               "padding": "6px 10px", "borderBottom": "1px solid #eee",
               "flexWrap": "wrap", "flex": "0 0 auto"},
        children=[
            _bar_item("Architecture", _any_dd(
                "at-explorer-arch-dd",
                [{"label": a["label"], "value": k} for k, a in ARCHITECTURES.items()],
                width="240px"),
                info="Model architecture used to turn trace token counts into "
                     "implied FLOPs, KV-cache bytes, and network traffic. Not "
                     "recorded in the traces — an assumption you pick. 'any' = "
                     f"{ARCHITECTURES[DEFAULT_ASSUMPTIONS['arch']]['label']}."),
            _bar_item("GPU", _any_dd(
                "at-explorer-gpu-dd",
                [{"label": g["label"], "value": k} for k, g in GPUS.items()],
                width="120px"),
                info="GPU whose spec-sheet peak FLOPs and HBM bandwidth "
                     "convert the implied work into GPU-seconds and GPU "
                     "counts. 'any' = "
                     f"{GPUS[DEFAULT_ASSUMPTIONS['gpu']]['label']}."),
            _bar_item("Weights q", _any_dd(
                "at-explorer-wdtype-dd",
                [{"label": d, "value": d} for d in ("bf16", "fp8")], width="90px"),
                info="Numeric precision of the model weights (bf16 = 2 bytes/"
                     "param, fp8 = 1). Sets weight-read bytes and which peak-"
                     "FLOPs spec applies. 'any' = "
                     f"{DEFAULT_ASSUMPTIONS['wdtype']}."),
            _bar_item("KV q", _any_dd(
                "at-explorer-kvdtype-dd",
                [{"label": d, "value": d} for d in ("bf16", "fp8")], width="90px"),
                info="Numeric precision of the KV cache (bf16 = 2 bytes/entry, "
                     "fp8 = 1). Sets KV read/write bytes and footprint. 'any' "
                     f"= {DEFAULT_ASSUMPTIONS['kvdtype']}."),
            _bar_item("TP", _any_dd("at-explorer-tp-dd",
                                    _num_options([1, 2, 4, 8, 16, 32]), width="72px"),
                      info="Tensor parallelism — GPUs splitting each layer's "
                           "matmuls; adds all-reduce traffic per token. 'any' "
                           f"= {DEFAULT_ASSUMPTIONS['tp']}."),
            _bar_item("PP", _any_dd("at-explorer-pp-dd",
                                    _num_options([1, 2, 4, 8]), width="72px"),
                      info="Pipeline parallelism — layers split into "
                           "sequential GPU stages; adds activation traffic at "
                           "each stage boundary. 'any' = "
                           f"{DEFAULT_ASSUMPTIONS['pp']}."),
            _bar_item("DP", _any_dd("at-explorer-dp-dd",
                                    _num_options([1, 2, 4, 8, 16, 32]), width="72px"),
                      info="Data parallelism — independent model replicas "
                           "serving different requests. No per-request "
                           "traffic; carried for cluster sizing. 'any' = "
                           f"{DEFAULT_ASSUMPTIONS['dp']}."),
            _bar_item("MFU %", _any_dd("at-explorer-mfu-dd",
                                       _num_options([20, 30, 40, 50, 60, 70, 80]),
                                       width="80px"),
                      info="Model FLOPs Utilization — fraction of the GPU's "
                           "peak tensor FLOPs actually sustained; divides "
                           "into implied compute time. 'any' = "
                           f"{DEFAULT_ASSUMPTIONS['mfu']}%."),
            _bar_item("MBU %", _any_dd("at-explorer-mbu-dd",
                                       _num_options([30, 40, 50, 60, 70, 80, 90]),
                                       width="80px"),
                      info="Memory Bandwidth Utilization — fraction of peak "
                           "HBM bandwidth actually sustained; divides into "
                           "implied memory-movement time. 'any' = "
                           f"{DEFAULT_ASSUMPTIONS['mbu']}%."),
        ],
    )

    center = html.Div(
        style={"flex": "1 1 0%", "minWidth": "0", "display": "flex",
               "flexDirection": "column", "overflow": "hidden"},
        children=[
            assumption_bar,
            dcc.Graph(id="at-explorer-growth-graph", config=GRAPH_CONFIG,
                      style={"flex": "1 1 auto", "minHeight": "0"}),
            html.Div(
                style={"flex": "0 0 32%", "minHeight": "0", "overflow": "hidden",
                       "borderTop": "1px solid #ddd", "padding": "4px 10px 8px",
                       "display": "flex", "flexDirection": "column"},
                children=[
                    dash_table.DataTable(
                        id="at-explorer-conv-table",
                        columns=TABLE_COLUMNS,
                        data=[],
                        **TABLE_TOOLTIP_KWARGS,
                        sort_action="native",
                        sort_mode="multi",
                        filter_action="native",
                        row_selectable="multi",
                        selected_rows=[],
                        page_action="none",
                        fixed_rows={"headers": True},
                        style_table={"height": "100%", "overflowY": "auto",
                                     "overflowX": "auto"},
                        style_cell={"fontFamily": MONO, "fontSize": "12px",
                                    "textAlign": "right", "padding": "2px 8px",
                                    "minWidth": "58px", "maxWidth": "230px",
                                    "whiteSpace": "nowrap", "overflow": "hidden",
                                    "textOverflow": "ellipsis"},
                        style_header={"fontWeight": "700", "background": "#f2f2f2"},
                        style_filter={"background": "#fbfbf3"},
                    ),
                ],
            ),
        ],
    )

    stores = [
        dcc.Store(id="at-explorer-filter-store"),
        dcc.Store(id="at-explorer-selection-store"),
        dcc.Store(id="at-explorer-config-store"),
        dcc.Store(id="at-explorer-visible-store"),
        dcc.Store(id="at-explorer-chartopts-store"),
        dcc.Store(id="at-explorer-sort-store"),
    ]
    return html.Div(id="at-explorer-tab", style={"height": "100%"},
                    children=[controls.tab_root(side, center, stores)])
