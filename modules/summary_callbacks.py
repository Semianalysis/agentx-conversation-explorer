"""Overview tab callbacks: import datasets, bulk-download traces, render cards."""
from __future__ import annotations

import logging

from dash import ALL, Input, Output, State, callback_context, html, no_update
from dash.exceptions import PreventUpdate

import base64
import tempfile
from pathlib import Path

from modules import api_client, controls, hf_client, records
from modules.controls import info
from modules.explorer_data import apply_selection
from modules.theme import F_SMALL, MONO, color_for, fmt_count

logger = logging.getLogger(__name__)

# Hover-help for card stats, keyed by stat label (one source of truth for the
# dataset cards AND the selection summary, which show the same stats).
STAT_INFO = {
    "Main-agent turns": "API requests made by the top-level agent of each "
                        "conversation.",
    "Subagent turns": "API requests made by nested agents spawned via tool "
                      "calls — each runs with its own context.",
    "Subagent groups": "Bursts of nested-agent activity: one group = one "
                       "subagent tree spawned from a main-agent turn.",
    "Input tokens": "Total input (context) tokens across all requests, cached "
                    "+ uncached. Each request re-reads its whole context, so "
                    "this grows quadratically with conversation length.",
    "Output tokens": "Total decoded (generated) tokens across all requests.",
    "Cached fraction": "Share of all input tokens served from the prompt "
                       "cache (cached ÷ total input). High = most context "
                       "re-reads are cache hits, not recomputed prefill.",
    "Median reqs/conv": "Requests per conversation, counting main-agent AND "
                        "subagent requests.",
    "Mean reqs/conv": "Requests per conversation, counting main-agent AND "
                      "subagent requests.",
    "Cache block size": "Prompt-cache granularity — caching happens in blocks "
                        "of this many tokens.",
    "Conversations": None,
}


def _stat(label: str, value: str) -> html.Div:
    tip = STAT_INFO.get(label)
    return html.Div(
        style={"display": "flex", "justifyContent": "space-between", "gap": "16px"},
        children=[
            html.Span([label, info(tip)] if tip else label,
                      style={"color": "#666"}),
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
                         html.Span("imported locally (internal source)",
                                   style={"fontSize": F_SMALL, "color": "#a60"})
                         if detail.get("source") == "hf" else
                         html.Button(
                             "Download traces" if n_cached < n_convs else "Re-check traces",
                             id={"type": "at-summary-download-btn", "slug": slug},
                             n_clicks=0,
                             title="Download every conversation trace of this "
                                   "dataset into the local data/ cache "
                                   "(already-cached ones are skipped). The "
                                   "other tabs read only this cache.",
                             style={"fontSize": F_SMALL, "padding": "5px 12px"}),
                         html.Span(dl_label, style={"fontSize": F_SMALL, "fontFamily": MONO,
                                                    "color": "#2a7" if n_cached >= n_convs and n_convs else "#a60"}),
                         html.A("HuggingFace ↗", href=detail.get("hf_url") or "#",
                                target="_blank", style={"fontSize": F_SMALL,
                                                        "marginLeft": "auto"}),
                     ]),
        ],
    )


def _preview_card(detail: dict, where: str) -> html.Div:
    """Stats for a dataset we might want: its own summary, plus whether its
    traces are already on this machine."""
    s = detail.get("summary") or {}
    n_cached = len(api_client.cached_conversation_ids(detail["slug"]))
    n_convs = detail.get("conversation_count", 0)
    cached_pct = s.get("cachedPct")
    rows = [
        ("Where", where),
        ("Conversations", f"{n_convs:,}"),
        ("Cached here", f"{n_cached}/{n_convs}" if n_convs else str(n_cached)),
        ("Main-agent turns", f"{s.get('mainTurns', 0):,}"),
        ("Subagent turns", f"{s.get('subagentTurns', 0):,}"),
        ("Input tokens", fmt_count(s.get("totalIn", 0))),
        ("Output tokens", fmt_count(s.get("totalOut", 0))),
        ("Cached fraction",
         f"{cached_pct * 100:.2f}%" if cached_pct is not None else "n/a"),
    ]
    note = ("" if n_cached else
            " — no traces cached yet: load it, then use its card's "
            "Download traces button")
    return controls.card(f"Preview — {detail.get('label', detail['slug'])}",
                         rows,
                         info_text="The dataset's own reported stats" + note)


def register_summary_callbacks(app) -> None:
    @app.callback(
        Output("at-summary-datasets-store", "data"),
        Output("at-summary-import-status", "children"),
        Input("at-summary-import-btn", "n_clicks"),
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
        Output("at-summary-cache-store", "data"),
        Input({"type": "at-summary-download-btn", "slug": ALL}, "n_clicks"),
        State("at-summary-datasets-store", "data"),
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
        Output("at-summary-selection-summary", "children"),
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
        Output("at-summary-cards", "children"),
        Output("at-summary-loaded-title", "children"),
        Input("at-summary-datasets-store", "data"),
        Input("at-summary-cache-store", "data"),
        Input("at-sources-store", "data"),
    )
    def render_cards(datasets, cache_counts, session_sources):
        """Summary analyses the datasets this session has LOADED: published
        ones whose traces are cached, plus local imports the user brought
        in. A private import nobody loaded is not reported here."""
        try:
            from modules import sources as src
            wanted_local = set(src.slugs(session_sources or [], src.LOCAL))
            local = [d for d in hf_client.local_hf_datasets()
                     if d["slug"] in wanted_local]
            website = [d for d in (datasets or [])
                       if api_client.cached_conversation_ids(d["slug"])]
            datasets = website + local
            title = (f"Loaded datasets ({len(datasets)})" if datasets
                     else "Loaded datasets")
            if not datasets:
                return html.Div(
                    "Nothing loaded yet — find a dataset above, or load a "
                    "local export from the Dataset panel on Explorer.",
                    style={"color": "#777", "fontSize": "15px",
                           "padding": "20px"}), title
            cache_counts = cache_counts or {}
            return [_dataset_card(
                d, cache_counts.get(d["slug"],
                                    len(api_client.cached_conversation_ids(d["slug"]))))
                for d in datasets], title
        except Exception:
            logger.exception("overview card render failed")
            raise PreventUpdate

    @app.callback(
        Output("at-summary-found-list", "children"),
        Output("at-summary-found-store", "data"),
        Output("at-summary-preview", "children"),
        Input("at-summary-search-btn", "n_clicks"),
        Input("at-summary-search-input", "value"),
        Input("at-summary-scope-radio", "value"),
        prevent_initial_call=True,
    )
    def find_datasets(_n, query, scope):
        """Search published datasets or local imports by name and preview
        the best match's own stats. Previewing reads dataset cards only - it
        never loads traces."""
        try:
            q = (query or "").strip().lower()
            if scope == "local":
                pool = hf_client.local_hf_datasets()
                where = "imported on this machine"
            else:
                pool = list(api_client.fetch_datasets())
                where = "published on the AgentX site"
            hits = [d for d in pool
                    if q in d["slug"].lower()
                    or q in str(d.get("label", "")).lower()] if q else pool
            if not hits:
                return (html.Div(f"nothing {where} matches {query!r}",
                                 style={"fontSize": F_SMALL, "color": "#a60"}),
                        None, "")
            rows = [html.Div(
                f"{d.get('label', d['slug'])}  —  {d['slug']}  "
                f"({d.get('conversation_count', '?')} conversations)",
                style={"fontSize": "12px", "fontFamily": MONO,
                       "color": "#333" if i == 0 else "#888"})
                for i, d in enumerate(hits[:8])]
            if len(hits) > 8:
                rows.append(html.Div(f"… and {len(hits) - 8} more",
                                     style={"fontSize": "11px",
                                            "color": "#999"}))
            best = hits[0]
            detail = best
            if scope == "online":
                try:
                    detail = api_client.fetch_dataset_detail(best["slug"])
                except Exception:
                    logger.exception("preview detail fetch failed")
            return (html.Div(rows, style={"margin": "6px 0"}),
                    {"detail": detail}, _preview_card(detail, where))
        except Exception:
            logger.exception("dataset search failed")
            return (html.Div("search FAILED — see server log",
                             style={"fontSize": F_SMALL, "color": "#a33"}),
                    None, "")

    if not hf_client.internal_mode():
        return  # internal-source callbacks reference components that only
        # exist in internal mode — never registered in the public build

    @app.callback(
        Output("at-summary-hf-select-cl", "options"),
        Output("at-summary-hf-select-cl", "value"),
        Output("at-summary-hf-unsupported", "children"),
        Input("at-summary-hf-list-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def list_internal(_n):
        try:
            rows = hf_client.list_source_datasets()
            if not rows:
                return [], [], html.Div(
                    "no datasets visible for "
                    f"{', '.join(hf_client.hf_sources())}",
                    style={"fontSize": F_SMALL, "color": "#999"})
            opts, rows_out = [], []
            for d in rows:
                badge = "PRIVATE" if d["private"] else "public"
                extra = (f"  [{badge}, {d['updated']}"
                         + (f", PARTIAL {d['imported_convs']} convs - will resume"
                            if d.get("resumable")
                            else (f", imported {d['imported_convs']} convs"
                                  if d["imported_convs"] else ""))
                         + "]")
                if d["importable"]:
                    opts.append({"label": " " + d["id"] + extra,
                                 "value": d["id"]})
                else:
                    rows_out.append(html.Div(
                        d["id"] + extra + "  - schema not supported",
                        style={"fontSize": "12px", "fontFamily": MONO,
                               "color": "#999", "padding": "1px 0 1px 22px"}))
            return opts, [], rows_out
        except Exception:
            logger.exception("internal dataset listing failed")
            return no_update, no_update, html.Div(
                "listing FAILED — see server log (is HF_TOKEN valid?)",
                style={"fontSize": F_SMALL, "color": "#a33"})

    @app.callback(
        Output("at-summary-hf-status", "children"),
        Input("at-summary-hf-load-btn", "n_clicks"),
        State("at-summary-hf-select-cl", "value"),
        prevent_initial_call=True,
    )
    def start_load(n, selected):
        try:
            if not n:
                raise PreventUpdate
            hf_client.start_background_import(selected or [])
            return f"loading {len(selected)} dataset(s) in the background..."
        except PreventUpdate:
            raise
        except ValueError as e:
            return str(e)
        except Exception:
            logger.exception("background import start failed")
            return "start FAILED - see server log"

    @app.callback(
        Output("at-summary-hf-progress", "children"),
        Output("at-summary-cache-store", "data", allow_duplicate=True),
        Input("at-summary-hf-interval", "n_intervals"),
        State("at-summary-cache-store", "data"),
        prevent_initial_call=True,
    )
    def poll_progress(_n, cache_counts):
        """1 Hz poll of the background importer: progress bar while running,
        Done/FAILED message after; refreshes the cache-store once when a
        finished import marks state dirty (dropdowns + cards update)."""
        try:
            p = hf_client.import_progress()
            cache_update = no_update
            if hf_client.consume_dirty():
                records.clear_pools()
                cache_counts = dict(cache_counts or {})
                for d in hf_client.local_hf_datasets():
                    cache_counts[d["slug"]] = d["conversation_count"]
                cache_update = cache_counts
            if p["running"]:
                done, total = p["done_convs"], p["total_convs"]
                pct = int(100 * done / total) if total else 0
                bar = html.Div(
                    style={"width": "420px", "height": "10px",
                           "background": "#eee", "borderRadius": "4px"},
                    children=html.Div(style={
                        "width": f"{pct}%", "height": "100%",
                        "background": "#2a7", "borderRadius": "4px"}))
                label = (f"{p['current']}: {done}/{total or '?'} conversations"
                         f" - {len(p['done_datasets'])} dataset(s) finished, "
                         f"{len(p['queue'])} queued")
                return html.Div([bar, html.Div(label, style={
                    "fontSize": "12px", "fontFamily": MONO})]), cache_update
            if p["error"]:
                return html.Div("FAILED: " + p["error"],
                                style={"fontSize": "12px", "color": "#a33"}), \
                    cache_update
            if p["done_datasets"]:
                return html.Div(
                    f"Done - imported {len(p['done_datasets'])} dataset(s): "
                    + ", ".join(p["done_datasets"]),
                    style={"fontSize": "12px", "color": "#2a7",
                           "fontWeight": "600"}), cache_update
            if cache_update is no_update:
                raise PreventUpdate
            return no_update, cache_update
        except PreventUpdate:
            raise
        except Exception:
            logger.exception("progress poll failed")
            raise PreventUpdate
