"""Explorer tab layout: growth-curve chart (most of the screen) + scrollable,
sortable, filterable conversation list.

The list — via column sorting (click; clicks stack for 2nd/3rd-order sort)
and the native per-column filter row (supports ranges: >100000, <=2) — is the
main filtering surface; it shows TRACE FACTS only (hardware cost estimates
were removed 2026-08-30: proprietary-model weights/data-movement guessing is
left to simulation via the Deep-dive sweep export). The left column keeps
only dataset + selection controls.

Cross-tab stores live HERE (all tabs stay mounted, see app.py):
  at-explorer-filter-store     {'slug'}
  at-explorer-selection-store  {'slug', 'conv_ids': [...]}
  at-explorer-config-store     serving assumptions (None = 'any')
  at-explorer-visible-store    conv_ids surviving the list's native filters
"""
from __future__ import annotations

from dash import dash_table, dcc, html

from modules import controls
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
}

# Kwargs shared by both DataTables so header help can never diverge.
TABLE_TOOLTIP_KWARGS = dict(tooltip_header=COLUMN_TOOLTIPS, tooltip_delay=400,
                            tooltip_duration=None)


def layout() -> html.Div:
    side = controls.sidebar([
        controls.label("Dataset",
                       info_text="The dataset is SHARED state: every tab is a "
                                 "viewport onto the same data, and picking a "
                                 "dataset here switches all of them. The list "
                                 "offers the PUBLISHED datasets by default; a "
                                 "dataset imported from this machine appears "
                                 "only once you load it below, so private "
                                 "traces are never surfaced on their own. "
                                 "Browse and preview more on the Summary tab."),
        controls.dropdown("at-explorer-dataset-dd", "pick a dataset"),
        html.Div(style={"display": "flex", "gap": "6px", "flexWrap": "wrap",
                        "margin": "6px 0 2px", "alignItems": "center"},
                 children=[
                     dcc.Upload(
                         id="at-explorer-load-local",
                         accept=".zip", multiple=False,
                         children=html.Button(
                             "Load local export…",
                             title="Pick an agent-trace export (.zip) from "
                                   "this machine. It is flattened into the "
                                   "gitignored local cache and added to this "
                                   "session's sources - it never leaves the "
                                   "machine.",
                             style={"fontSize": "11px", "padding": "3px 8px"})),
                     dcc.Upload(
                         id="at-explorer-load-session",
                         accept=".json", multiple=False,
                         children=html.Button(
                             "Import session…",
                             title="Restore a source list exported earlier: "
                                   "the datasets it names are re-activated "
                                   "for this session. Sources whose cache is "
                                   "missing on this machine are reported, not "
                                   "silently skipped.",
                             style={"fontSize": "11px", "padding": "3px 8px"})),
                     html.Button("Export session",
                                 id="at-explorer-export-session-btn", n_clicks=0,
                                 title="Download this session's source list "
                                       "(slugs, labels, and where local "
                                       "exports came from - never trace data) "
                                       "so tomorrow starts where today ended.",
                                 style={"fontSize": "11px", "padding": "3px 8px"}),
                 ]),
        html.Div(id="at-explorer-sources-status",
                 style={"fontSize": "11px", "color": "#666",
                        "whiteSpace": "pre-wrap", "marginBottom": "4px"}),
        controls.label("Chart x measure",
                       info_text="X axis of the growth chart. turn count = "
                                 "main-agent turn ordinal (1..n). cumulative "
                                 "time = wall-clock seconds since the "
                                 "conversation's first request — idle "
                                 "stretches (human think time, nights) show "
                                 "as flat gaps. busy time = only ACTIVE "
                                 "seconds, when at least one request (main "
                                 "or subagent) was in flight — idle gaps "
                                 "compressed out. Time values below 1 s are "
                                 "shown at 1."),
        dcc.RadioItems(
            id="at-explorer-xmeasure-radio",
            options=[{"label": " turn count", "value": "turn"},
                     {"label": " cumulative time", "value": "cumulative_time"},
                     {"label": " busy time", "value": "busy_time"}],
            value="turn", style={"fontSize": F_SMALL}),
        controls.label("Chart x scale",
                       info_text="Scale of the growth chart's x axis. Log "
                                 "compresses very long conversations so "
                                 "short ones stay readable."),
        dcc.RadioItems(
            id="at-explorer-xscale-radio",
            options=[{"label": " linear", "value": "linear"},
                     {"label": " log", "value": "log"}],
            value="log", style={"fontSize": F_SMALL}),
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
        html.Div("The list shows trace facts only (tokens, turns, timing). "
                 "Hardware cost estimates are deliberately not shown — use "
                 "the Deep-dive sweep export and simulate.",
                 style={"fontSize": "11px", "color": "#999", "marginTop": "10px"}),
    ])

    center = html.Div(
        style={"flex": "1 1 0%", "minWidth": "0", "display": "flex",
               "flexDirection": "column", "overflow": "hidden"},
        children=[
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
