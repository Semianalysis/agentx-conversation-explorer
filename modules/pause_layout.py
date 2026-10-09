"""Pause Analytics tab layout.

A pause is dead air on a conversation's main-agent lane. Two charts share
ONE duration axis (the same bin edges), so a bar of stacked causes lines up
with the ISL that arrived when that pause ended. All Pause dcc.Stores live
here; the dataset and the conversation selection come from the shared
cross-tab stores, everything else here is a viewport option.
"""
from __future__ import annotations

from dash import dcc, html

from modules import controls
from modules.theme import F_SMALL, GRAPH_CONFIG

ISL_MEASURES = {
    "next_in_tokens": "total context (ISL) of the next turn",
    "next_uncached_tokens": "uncached input delivered after the pause",
    "next_cached_tokens": "cached prefix at the next turn",
    "next_out_tokens": "decode output of the next turn",
}


def layout() -> html.Div:
    side = controls.sidebar([
        controls.label("Dataset",
                       info_text="Shared with every tab: picking a dataset "
                                 "here switches all of them. An Explorer "
                                 "conversation selection also scopes these "
                                 "pauses."),
        controls.dropdown("at-pause-dataset-dd", "pick a cached dataset"),
        controls.label("Minimum pause (s)",
                       info_text="Ignore pauses shorter than this. Sub-second "
                                 "gaps between back-to-back requests are "
                                 "scheduling noise, not waiting; raise this "
                                 "to look only at real waits."),
        dcc.Input(id="at-pause-min-input", type="number", value=1,
                  min=0, step=0.5, debounce=True,
                  style={"width": "100%", "fontSize": F_SMALL}),
        controls.label("Duration bins",
                       info_text="Scale of the shared duration axis. Log bins "
                                 "are right for pauses spanning milliseconds "
                                 "to days; both charts use these exact edges "
                                 "so they stay aligned."),
        dcc.RadioItems(
            id="at-pause-scale-radio",
            options=[{"label": " log bins", "value": "log"},
                     {"label": " linear bins", "value": "linear"}],
            value="log", style={"fontSize": F_SMALL}),
        controls.label("Stack measures",
                       info_text="time = seconds of each cause inside the "
                                 "pauses of that duration (bar height is real "
                                 "waiting). count = number of pauses, stacked "
                                 "by their LARGEST cause."),
        dcc.RadioItems(
            id="at-pause-weight-radio",
            options=[{"label": " pause time", "value": "time"},
                     {"label": " pause count", "value": "count"}],
            value="time", style={"fontSize": F_SMALL}),
        controls.label("Causes (stacked, selectable)",
                       info_text="What was running during the pause. "
                                 "'subagent' comes from the trace structure "
                                 "itself; tool categories come from an "
                                 "imported export that recorded them; "
                                 "'unknown' is the honest remainder - human "
                                 "think time, queueing, anything the data "
                                 "does not record. Untick a cause to drop it "
                                 "from the stack."),
        dcc.Checklist(id="at-pause-causes-cl", options=[], value=[],
                      style={"fontSize": F_SMALL},
                      labelStyle={"display": "block"}),
        controls.label("Second chart measure",
                       info_text="What the request that ENDED each pause "
                                 "delivered - plotted against the same "
                                 "duration bins, so you can see whether long "
                                 "waits arrive with bigger prompts."),
        dcc.RadioItems(
            id="at-pause-isl-radio",
            options=[{"label": " " + lbl, "value": k}
                     for k, lbl in ISL_MEASURES.items()],
            value="next_in_tokens", style={"fontSize": F_SMALL}),
        html.Div(id="at-pause-gate-status",
                 style={"fontSize": "12px", "color": "#666", "marginTop": "16px",
                        "fontFamily": "monospace", "whiteSpace": "pre-wrap"}),
    ])

    center = html.Div(
        style={"flex": "1 1 0%", "minWidth": "0", "display": "flex",
               "flexDirection": "column", "overflow": "hidden",
               "padding": "4px 10px", "gap": "2px"},
        children=[
            html.Div(dcc.Graph(id="at-pause-stack-graph", config=GRAPH_CONFIG,
                               style={"height": "100%", "width": "100%"}),
                     style={"flex": "1 1 0%", "minHeight": "0"}),
            html.Div(dcc.Graph(id="at-pause-isl-graph", config=GRAPH_CONFIG,
                               style={"height": "100%", "width": "100%"}),
                     style={"flex": "1 1 0%", "minHeight": "0"}),
        ],
    )

    right = html.Div(
        id="at-pause-summary-panel",
        style={"flex": "0 0 19%", "minWidth": "260px", "maxWidth": "19%",
               "overflowY": "auto", "padding": "12px",
               "borderLeft": "1px solid #ddd", "background": "#fcfcfc"},
    )

    stores = [dcc.Store(id="at-pause-opts-store")]
    return html.Div(id="at-pause-tab", style={"height": "100%"},
                    children=[controls.tab_root(side, center, stores, right)])
