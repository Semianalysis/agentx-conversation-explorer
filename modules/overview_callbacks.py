"""Overview tab callbacks: import datasets, bulk-download traces, render cards."""
from __future__ import annotations

import logging

from dash import ALL, Input, Output, State, callback_context, html
from dash.exceptions import PreventUpdate

from modules import api_client, records
from modules.explorer_data import apply_selection
from modules.theme import F_SMALL, MONO, color_for, fmt_count

logger = logging.getLogger(__name__)


def _stat(label: str, value: str) -> html.Div:
    return html.Div(
        style={"display": "flex", "justifyContent": "space-between", "gap": "16px"},
        children=[
            html.Span(label, style={"color": "#666"}),
            html.Span(value, style={"fontFamily": MONO}),
        ],
    )


def _model_mix_rows(model_mix: dict) -> list[html.Div]:
    total = sum(model_mix.values())
    rows = []
    for model, n in sorted(model_mix.items(), key=lambda kv: -kv[1]):
        pct = n / total if total else 0
        rows.append(html.Div(
            style={"display": "flex", "alignItems": "center", "gap": "8px",
                   "fontSize": "12px", "fontFamily": MONO},
            children=[
                html.Span(model, style={"flex": "0 0 240px", "whiteSpace": "nowrap",
                                        "overflow": "hidden", "textOverflow": "ellipsis"}),
                html.Div(style={"flex": "1 1 auto", "background": "#eee",
                                "height": "10px", "borderRadius": "3px"},
                         children=html.Div(style={
                             "width": f"{pct * 100:.1f}%", "height": "100%",
                             "background": color_for(model), "borderRadius": "3px"})),
                html.Span(f"{n:,} ({pct * 100:.1f}%)",
                          style={"flex": "0 0 130px", "textAlign": "right"}),
            ],
        ))
    return rows


def _dataset_card(detail: dict, n_cached: int) -> html.Div:
    slug = detail["slug"]
    s = detail.get("summary") or {}
    n_convs = detail.get("conversation_count", 0)
    total_in = s.get("totalIn", 0)
    total_out = s.get("totalOut", 0)
    cached_pct = s.get("cachedPct")
    dl_label = (f"Traces cached: {n_cached}/{n_convs}" if n_cached
                else "No traces cached yet")
    return html.Div(
        style={"border": "1px solid #ccc", "borderRadius": "8px", "padding": "14px",
               "width": "560px", "background": "white",
               "boxShadow": "0 1px 3px rgba(0,0,0,0.08)"},
        children=[
            html.Div(style={"display": "flex", "alignItems": "baseline", "gap": "10px"},
                     children=[
                         html.H3(detail.get("label", slug), style={"margin": "0"}),
                         html.Span(detail.get("variant") or "", style={"color": "#888",
                                                                       "fontSize": F_SMALL}),
                     ]),
            html.Div(detail.get("description") or "", style={"fontSize": F_SMALL,
                                                             "color": "#555",
                                                             "margin": "6px 0 10px"}),
            html.Div(style={"display": "grid",
                            "gridTemplateColumns": "1fr 1fr", "gap": "4px 24px",
                            "fontSize": F_SMALL, "marginBottom": "10px"},
                     children=[
                         _stat("Conversations", f"{n_convs:,}"),
                         _stat("Main-agent turns", f"{s.get('mainTurns', 0):,}"),
                         _stat("Subagent turns", f"{s.get('subagentTurns', 0):,}"),
                         _stat("Subagent groups", f"{s.get('subagentGroups', 0):,}"),
                         _stat("Input tokens", fmt_count(total_in)),
                         _stat("Output tokens", fmt_count(total_out)),
                         _stat("Cached fraction",
                               f"{cached_pct * 100:.2f}%" if cached_pct is not None else "n/a"),
                         _stat("Median reqs/conv",
                               f"{s.get('medianRequestsPerConversation', 0):,}"),
                         _stat("Mean reqs/conv",
                               f"{s.get('meanRequestsPerConversation', 0):.1f}"),
                         _stat("Cache block size", f"{s.get('blockSize', '?')} tokens"),
                     ]),
            html.Div("Model mix (requests per model)",
                     style={"fontSize": F_SMALL, "fontWeight": "600", "margin": "6px 0"}),
            html.Div(_model_mix_rows(s.get("modelMix") or {}),
                     style={"display": "flex", "flexDirection": "column", "gap": "3px"}),
            html.Div(style={"display": "flex", "alignItems": "center", "gap": "12px",
                            "marginTop": "12px"},
                     children=[
                         html.Button(
                             "Download traces" if n_cached < n_convs else "Re-check traces",
                             id={"type": "at-overview-download-btn", "slug": slug},
                             n_clicks=0, style={"fontSize": F_SMALL, "padding": "5px 12px"}),
                         html.Span(dl_label, style={"fontSize": F_SMALL, "fontFamily": MONO,
                                                    "color": "#2a7" if n_cached >= n_convs and n_convs else "#a60"}),
                         html.A("HuggingFace ↗", href=detail.get("hf_url") or "#",
                                target="_blank", style={"fontSize": F_SMALL,
                                                        "marginLeft": "auto"}),
                     ]),
        ],
    )


def register_overview_callbacks(app) -> None:
    @app.callback(
        Output("at-overview-datasets-store", "data"),
        Output("at-overview-import-status", "children"),
        Input("at-overview-import-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def import_datasets(n_clicks):
        try:
            registry = api_client.fetch_datasets(refresh=True)
            details = [api_client.fetch_dataset_detail(d["slug"], refresh=True)
                       for d in registry]
            for d in details:
                api_client.fetch_conversation_index(d["slug"], refresh=True)
            n_reqs = sum((d.get("summary") or {}).get("mainTurns", 0)
                         + (d.get("summary") or {}).get("subagentTurns", 0)
                         for d in details)
            status = (f"Imported {len(details)} datasets, "
                      f"{sum(d.get('conversation_count', 0) for d in details):,} conversations, "
                      f"{n_reqs:,} requests total")
            return details, status
        except Exception:
            logger.exception("dataset import failed")
            return [], "Import FAILED — see server log"

    @app.callback(
        Output("at-overview-cache-store", "data"),
        Input({"type": "at-overview-download-btn", "slug": ALL}, "n_clicks"),
        State("at-overview-datasets-store", "data"),
        prevent_initial_call=True,
    )
    def download_traces(n_clicks_list, datasets):
        try:
            trig = callback_context.triggered_id
            if not isinstance(trig, dict) or not any(
                    t.get("value") for t in callback_context.triggered):
                raise PreventUpdate
            slug = trig["slug"]
            summary = api_client.bulk_download_conversations(slug)
            logger.info("download for %s: %s", slug, summary)
            if summary["failed"]:
                logger.error("download failures for %s: %s", slug, summary["errors"])
            records.clear_pools()  # cache changed; pools must reload from disk
            return {d["slug"]: len(api_client.cached_conversation_ids(d["slug"]))
                    for d in (datasets or [])}
        except PreventUpdate:
            raise
        except Exception:
            logger.exception("trace download failed")
            raise PreventUpdate

    @app.callback(
        Output("at-overview-selection-summary", "children"),
        Input("at-explorer-selection-store", "data"),
    )
    def render_selection_summary(selection):
        """When the Explorer has a conversation selection, summarize JUST that
        subset (computed from the cached records, same fields as the dataset
        cards) above the full-dataset cards."""
        try:
            slug = (selection or {}).get("slug")
            conv_ids = (selection or {}).get("conv_ids") or []
            if not slug or not conv_ids:
                return None
            pool = records.load_records(slug)
            subset, _ = apply_selection(pool, selection, slug)
            if not subset:
                return None
            n_main = sum(1 for r in subset if r["role"] == "main")
            total_in = sum(r["in_tokens"] for r in subset)
            total_cached = sum(r["cached_tokens"] for r in subset)
            total_out = sum(r["out_tokens"] for r in subset)
            mix: dict[str, int] = {}
            for r in subset:
                mix[r["model"]] = mix.get(r["model"], 0) + 1
            return html.Div(
                style={"border": "2px solid #1f77b4", "borderRadius": "8px",
                       "padding": "14px", "marginBottom": "14px", "background": "#f4f9ff",
                       "maxWidth": "1134px"},
                children=[
                    html.H3(f"Explorer selection — {len(conv_ids)} conversations "
                            f"of {slug}", style={"margin": "0 0 8px"}),
                    html.Div(style={"display": "grid",
                                    "gridTemplateColumns": "1fr 1fr 1fr",
                                    "gap": "4px 24px", "fontSize": F_SMALL,
                                    "marginBottom": "10px"},
                             children=[
                                 _stat("Requests", f"{len(subset):,}"),
                                 _stat("Main-agent turns", f"{n_main:,}"),
                                 _stat("Subagent turns", f"{len(subset) - n_main:,}"),
                                 _stat("Input tokens", fmt_count(total_in)),
                                 _stat("Output tokens", fmt_count(total_out)),
                                 _stat("Cached fraction",
                                       f"{total_cached / total_in * 100:.2f}%"
                                       if total_in else "n/a"),
                             ]),
                    html.Div("Model mix (requests per model)",
                             style={"fontSize": F_SMALL, "fontWeight": "600",
                                    "margin": "6px 0"}),
                    html.Div(_model_mix_rows(mix),
                             style={"display": "flex", "flexDirection": "column",
                                    "gap": "3px"}),
                    html.Div("Turns and Correlations are now scoped to this selection; "
                             "full-dataset cards below.",
                             style={"fontSize": "11px", "color": "#777",
                                    "marginTop": "8px"}),
                ],
            )
        except Exception:
            logger.exception("selection summary render failed")
            raise PreventUpdate

    @app.callback(
        Output("at-overview-cards", "children"),
        Input("at-overview-datasets-store", "data"),
        Input("at-overview-cache-store", "data"),
    )
    def render_cards(datasets, cache_counts):
        try:
            if not datasets:
                return html.Div(
                    "No datasets imported yet — click 'Import / refresh datasets'.",
                    style={"color": "#777", "fontSize": "15px", "padding": "20px"})
            cache_counts = cache_counts or {}
            return [_dataset_card(d, cache_counts.get(d["slug"], 0)) for d in datasets]
        except Exception:
            logger.exception("overview card render failed")
            raise PreventUpdate
