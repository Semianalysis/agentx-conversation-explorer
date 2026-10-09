"""Explorer tab callbacks: growth-curve chart, conversation table, and the
cross-tab conversation selection.
"""
from __future__ import annotations

import base64
import json
import logging
import tempfile
from pathlib import Path

import plotly.graph_objects as go
from dash import Input, Output, State, callback_context, no_update
from dash.exceptions import PreventUpdate

from modules import api_client, records, theme
from modules.explorer_data import (CURVE_X_MEASURES,
                                   build_conversation_table,
                                   conversation_curves, sync_updates)
from modules.explorer_layout import TABLE_COLUMNS
from modules.theme import color_for

logger = logging.getLogger(__name__)


def _decode_upload(contents: str) -> bytes:
    """dcc.Upload 'data:<mime>;base64,<payload>' -> raw bytes."""
    _, _, b64 = contents.partition(",")
    return base64.b64decode(b64)


def _write_upload(contents: str, filename: str) -> Path:
    path = Path(tempfile.gettempdir()) / filename
    path.write_bytes(_decode_upload(contents))
    return path


def _available_slugs() -> set[str]:
    """Datasets this machine can actually open right now."""
    return {p.parent.name for p in api_client.DATA_DIR.glob("*/detail.json")}

# The dataset is GLOBAL state — every tab is a viewport onto the same data.
# These dropdowns are views of ONE value, synced both ways.
_DATASET_DDS = ("at-explorer-dataset-dd", "at-corr-dataset-dd",
                "at-pause-dataset-dd")

# Startup default: the FULL weka traces (user 2026-08-29: the 256k-limit
# variant is not very useful as a default). Falls back to the first cached
# dataset when the preferred one isn't on disk.
DEFAULT_DATASET_SLUG = "cc-traces-weka-062126"


def cached_dataset_options(session_sources: list[dict] | None = None
                           ) -> list[dict]:
    """Dropdown options for the shared dataset picker.

    PUBLISHED datasets (the AgentX registry) are always offered once their
    traces are cached. A LOCAL import is offered only when this session's
    source list names it, so private traces on this machine never appear by
    themselves — the user loads them from the Dataset panel or the Summary
    finder first.
    """
    from modules import hf_client, sources as src
    try:
        website = list(api_client.fetch_datasets())
    except Exception:
        website = []
    wanted_local = set(src.slugs(session_sources or [], src.LOCAL))
    local = [d for d in hf_client.local_hf_datasets()
             if d["slug"] in wanted_local]
    opts = []
    for d in website + local:
        n = len(api_client.cached_conversation_ids(d["slug"]))
        if n:
            mark = "" if src.source_kind(d) == src.WEBSITE else "  [local]"
            opts.append({"label": f"{d.get('label', d['slug'])}{mark} "
                                  f"({n} convs cached)",
                         "value": d["slug"]})
    return opts


def _annotate_sort_columns(sort_by: list[dict]) -> list[dict]:
    """Column defs with a broad arrow + priority numeral appended to sorted
    columns' names ('final ctx ▼2'). Numeral only when several columns sort."""
    order = {e["column_id"]: (i, e["direction"]) for i, e in enumerate(sort_by)}
    out = []
    for col in TABLE_COLUMNS:
        col = dict(col)
        if col["id"] in order:
            i, direction = order[col["id"]]
            arrow = "▲" if direction == "asc" else "▼"
            badge = f"{arrow}{i + 1}" if len(sort_by) > 1 else arrow
            col["name"] = f"{col['name']} {badge}"
        out.append(col)
    return out


def _register_sort_semantics(app, table_id: str, store_id: str) -> None:
    """Plain header click = THE sort (replaces any stack); ctrl-click = add to /
    adjust the multi-sort stack. DataTable's sort_mode='multi' always stacks, so
    this post-processes sort_by: assets/sort_ctrl.js records the ctrl-key state
    of the click into the hidden at-sort-ctrl-input just before it lands."""
    @app.callback(
        Output(table_id, "sort_by"),
        Output(table_id, "columns"),
        Output(store_id, "data"),
        Input(table_id, "sort_by"),
        State("at-sort-ctrl-input", "value"),
        State(store_id, "data"),
        prevent_initial_call=True,
    )
    def sort_semantics(sort_by, ctrl, prev):
        try:
            sort_by = sort_by or []
            prev = prev or []
            if sort_by == prev:
                raise PreventUpdate  # echo of our own write — stop the loop
            if ctrl != "1" and len(sort_by) > 1:
                prev_dirs = {e["column_id"]: e["direction"] for e in prev}
                new_ids = {e["column_id"] for e in sort_by}
                removed = [c for c in prev_dirs if c not in new_ids]
                changed = [e for e in sort_by
                           if e["column_id"] not in prev_dirs
                           or prev_dirs[e["column_id"]] != e["direction"]]
                if removed:
                    sort_by = []  # plain click cleared a column -> nothing sorted
                elif changed:
                    sort_by = [changed[-1]]
                else:
                    sort_by = sort_by[-1:]
            return sort_by, _annotate_sort_columns(sort_by), sort_by
        except PreventUpdate:
            raise
        except Exception:
            logger.exception("sort semantics failed for %s", table_id)
            raise PreventUpdate


def register_explorer_callbacks(app) -> None:
    _register_sort_semantics(app, "at-explorer-conv-table", "at-explorer-sort-store")
    _register_sort_semantics(app, "at-deep-conv-table", "at-deep-sort-store")

    @app.callback(
        [Output(dd, "value", allow_duplicate=True) for dd in _DATASET_DDS],
        [Input(dd, "value") for dd in _DATASET_DDS],
        prevent_initial_call=True,
    )
    def sync_dataset(*values):
        """Self-loop sync of the shared dataset across all tabs: whichever
        dropdown the user changed wins; the echo of our own write arrives with
        all values equal and stops the loop (sync_updates returns None)."""
        try:
            trig = callback_context.triggered_id
            if trig not in _DATASET_DDS:
                raise PreventUpdate
            updates = sync_updates(list(values), _DATASET_DDS.index(trig))
            if updates is None:
                raise PreventUpdate
            value, stale = updates
            return [value if i in stale else no_update
                    for i in range(len(_DATASET_DDS))]
        except PreventUpdate:
            raise
        except Exception:
            logger.exception("dataset sync failed")
            raise PreventUpdate

    @app.callback(
        Output("at-explorer-chartopts-store", "data"),
        Input("at-explorer-xscale-radio", "value"),
        Input("at-explorer-onlysel-cl", "value"),
        Input("at-explorer-xmeasure-radio", "value"),
    )
    def coalesce_chartopts(xscale, onlysel, xmeasure):
        return {"xscale": xscale, "only_selected": "only" in (onlysel or []),
                "xmeasure": xmeasure or "turn"}
    @app.callback(
        Output("at-explorer-dataset-dd", "options"),
        Output("at-explorer-dataset-dd", "value"),
        Input("at-tabs", "value"),
        Input("at-summary-cache-store", "data"),
        Input("at-sources-store", "data"),
        State("at-explorer-dataset-dd", "value"),
    )
    def dataset_options(_tab, _cache, session_sources, current):
        try:
            opts = cached_dataset_options(session_sources)
            slugs = {o["value"] for o in opts}
            if current in slugs:
                value = current
            elif DEFAULT_DATASET_SLUG in slugs:
                value = DEFAULT_DATASET_SLUG
            else:
                value = opts[0]["value"] if opts else None
            return opts, value
        except Exception:
            logger.exception("explorer dataset options failed")
            raise PreventUpdate

    @app.callback(
        Output("at-sources-store", "data"),
        Output("at-explorer-sources-status", "children"),
        Input("at-explorer-load-local", "contents"),
        Input("at-explorer-load-session", "contents"),
        Input("at-summary-load-btn", "n_clicks"),
        State("at-explorer-load-local", "filename"),
        State("at-explorer-load-session", "filename"),
        State("at-summary-found-store", "data"),
        State("at-sources-store", "data"),
        prevent_initial_call=True,
    )
    def manage_sources(local_contents, session_contents, _load_clicks,
                       local_name, session_name, found, current):
        """THE owner of the session source list: a local export loaded from
        the Dataset panel, a session file restored there, or a dataset the
        Summary finder loaded all converge here."""
        from modules import openclaw_import, sources as src
        trig = callback_context.triggered_id
        try:
            if trig == "at-explorer-load-local":
                if not local_contents:
                    raise PreventUpdate
                path = _write_upload(local_contents, local_name or "export.zip")
                try:
                    detail = openclaw_import.import_openclaw(path)
                finally:
                    path.unlink(missing_ok=True)
                records.clear_pools()
                out = src.add(current or [],
                              src.make_source(detail["slug"], src.LOCAL,
                                              detail.get("label", ""),
                                              local_name))
                return out, (f"loaded {detail['slug']}: "
                             f"{detail['conversation_count']} conversations")

            if trig == "at-explorer-load-session":
                if not session_contents:
                    raise PreventUpdate
                blob = json.loads(_decode_upload(session_contents))
                restored = src.parse_blob(blob)
                available = _available_slugs()
                present, missing = src.resolve(restored, available)
                msg = f"session restored: {len(present)} source(s)"
                if missing:
                    msg += ("\nnot on this machine: "
                            + ", ".join(m["slug"] for m in missing)
                            + " — load the export(s) again to re-add")
                return present, msg

            if trig == "at-summary-load-btn":
                detail = (found or {}).get("detail")
                if not detail:
                    raise PreventUpdate
                from modules import sources as s2
                out = s2.add(current or [],
                             s2.make_source(detail["slug"],
                                            s2.source_kind(detail),
                                            detail.get("label", "")))
                return out, f"added {detail['slug']} to this session"
            raise PreventUpdate
        except PreventUpdate:
            raise
        except Exception as e:
            logger.exception("source management failed")
            return no_update, f"FAILED: {e}"

    @app.callback(
        Output("at-sources-download", "data"),
        Input("at-explorer-export-session-btn", "n_clicks"),
        State("at-sources-store", "data"),
        prevent_initial_call=True,
    )
    def export_session(n, current):
        from dash import dcc as _dcc

        from modules import sources as src
        try:
            if not n:
                raise PreventUpdate
            blob = src.export_blob(current or [])
            return _dcc.send_string(json.dumps(blob, indent=1),
                                    "agentx-explorer-session.json")
        except PreventUpdate:
            raise
        except Exception:
            logger.exception("session export failed")
            raise PreventUpdate

    @app.callback(
        Output("at-explorer-filter-store", "data"),
        Input("at-explorer-dataset-dd", "value"),
    )
    def coalesce_filters(slug):
        return {"slug": slug}

    @app.callback(
        Output("at-explorer-conv-table", "data"),
        Output("at-deep-conv-table", "data"),
        Input("at-explorer-filter-store", "data"),
        prevent_initial_call=True,
    )
    def table_data(filters):
        """ONE conversation list served to both viewports (Explorer and
        Deep-dive) with identical rows in identical order, so row indices and
        the shared selection can never diverge."""
        try:
            if not filters or not filters.get("slug"):
                return [], []
            slug = filters["slug"]
            pool = records.load_records(slug)
            index_ids = [it["conv_id"] for it in api_client.fetch_conversation_index(slug)]
            rows, skipped = build_conversation_table(pool, index_ids, None)
            if skipped:
                logger.warning("%s: %d conversations without main turns skipped",
                               slug, skipped)
            return rows, rows
        except Exception:
            logger.exception("explorer table data failed")
            raise PreventUpdate

    @app.callback(
        Output("at-explorer-visible-store", "data"),
        Input("at-explorer-conv-table", "derived_virtual_row_ids"),
        State("at-explorer-visible-store", "data"),
        prevent_initial_call=True,
    )
    def visible_rows(derived_ids, current):
        """conv_ids surviving the list's native filter row. Sorting reorders
        derived_ids without changing membership — skip those echoes so the
        chart doesn't re-render on every header click."""
        new = sorted(x for x in (derived_ids or []) if x is not None)
        if current is not None and new == current:
            raise PreventUpdate
        return new

    @app.callback(
        Output("at-explorer-selection-store", "data"),
        Output("at-explorer-conv-table", "selected_rows"),
        Output("at-explorer-conv-table", "selected_cells"),
        Output("at-deep-conv-table", "selected_rows"),
        Output("at-deep-conv-table", "selected_cells"),
        Output("at-explorer-growth-graph", "clickData"),
        Output("at-explorer-selection-status", "children"),
        Input("at-explorer-conv-table", "selected_rows"),
        Input("at-explorer-conv-table", "selected_cells"),
        Input("at-deep-conv-table", "selected_rows"),
        Input("at-deep-conv-table", "selected_cells"),
        Input("at-explorer-growth-graph", "clickData"),
        Input("at-explorer-clear-btn", "n_clicks"),
        Input("at-explorer-filter-store", "data"),
        State("at-explorer-conv-table", "data"),
        State("at-explorer-selection-store", "data"),
        prevent_initial_call=True,
    )
    def pick_selection(ex_rows, ex_cells, dd_rows, dd_cells, click, _clear,
                       filters, table_rows, current):
        """THE selection callback: both viewports' checkboxes and cell drags,
        curve clicks, and the clear button all converge on the one selection
        store, and both tables are synced back from it — the two lists are the
        same object seen twice, so they can never diverge."""
        try:
            trig = callback_context.triggered_id
            slug = (filters or {}).get("slug")
            table_rows = table_rows or []
            current_ids = set((current or {}).get("conv_ids") or [])
            if (current or {}).get("slug") != slug:
                current_ids = set()

            if trig == "at-explorer-clear-btn":
                new_ids: set[str] = set()
            elif trig == "at-explorer-filter-store":
                # dataset switch invalidates the selection
                new_ids = current_ids if slug == (current or {}).get("slug") else set()
            elif trig == "at-explorer-growth-graph":
                if not click or not click.get("points"):
                    raise PreventUpdate
                cd = click["points"][0].get("customdata")
                if not cd:
                    raise PreventUpdate
                conv_id = cd[1]
                new_ids = current_ids ^ {conv_id}  # toggle
            else:  # checkboxes / cell drag in whichever viewport triggered
                sel_rows, sel_cells = ((dd_rows, dd_cells)
                                       if trig == "at-deep-conv-table"
                                       else (ex_rows, ex_cells))
                row_by_index = {i: r["id"] for i, r in enumerate(table_rows)}
                checkbox_ids = {row_by_index[i] for i in (sel_rows or [])
                                if i in row_by_index}
                cell_ids = {c["row_id"] for c in (sel_cells or []) if "row_id" in c}
                new_ids = checkbox_ids | cell_ids
                if new_ids == current_ids:
                    raise PreventUpdate  # echo of our own sync — stop the loop

            new_rows = [i for i, r in enumerate(table_rows) if r["id"] in new_ids]
            status = (f"selected: {len(new_ids)} conversations"
                      if new_ids else "no selection (all conversations)")
            store = {"slug": slug, "conv_ids": sorted(new_ids)}
            return store, new_rows, [], new_rows, [], None, status
        except PreventUpdate:
            raise
        except Exception:
            logger.exception("explorer selection failed")
            raise PreventUpdate

    @app.callback(
        Output("at-explorer-growth-graph", "figure"),
        Input("at-explorer-filter-store", "data"),
        Input("at-explorer-selection-store", "data"),
        Input("at-explorer-visible-store", "data"),
        Input("at-explorer-chartopts-store", "data"),
        prevent_initial_call=True,
    )
    def render_growth(filters, selection, visible_ids, chartopts):
        try:
            if not filters or not filters.get("slug"):
                fig = go.Figure()
                fig.update_layout(**theme.base_layout(
                    title="Conversation context growth"))
                fig.add_annotation(text="pick a dataset", showarrow=False,
                                   xref="paper", yref="paper", x=0.5, y=0.5)
                return fig
            slug = filters["slug"]
            opts = chartopts or {}
            pool = records.load_records(slug)
            index_ids = [it["conv_id"] for it in api_client.fetch_conversation_index(slug)]
            rows, _ = build_conversation_table(pool, index_ids, None)
            ordinal = {r["id"]: r["ordinal"] for r in rows}
            wanted = set(visible_ids) if visible_ids else set(ordinal)
            sel_ids = set((selection or {}).get("conv_ids") or []) \
                if (selection or {}).get("slug") == slug else set()
            if opts.get("only_selected") and sel_ids:
                wanted &= sel_ids
            x_measure = opts.get("xmeasure", "turn")
            curves = conversation_curves(pool, conv_ids=wanted & set(ordinal),
                                         x_measure=x_measure)
            return _growth_figure(curves, ordinal, sel_ids & set(curves), slug,
                                  log_x=opts.get("xscale") == "log",
                                  x_measure=x_measure)
        except Exception:
            logger.exception("explorer growth render failed")
            raise PreventUpdate


def _growth_figure(curves: dict, ordinal: dict, sel_ids: set, slug: str,
                   log_x: bool = False, x_measure: str = "turn") -> go.Figure:
    """One gray None-separated pool trace (fast) + a colored trace per selected
    conversation. customdata rows = [ordinal, conv_id] for click-to-select."""
    x_hover = ("turn %{x}" if x_measure == "turn" else "t=%{x:,.6g}s")
    fig = go.Figure()
    pool_x: list = []
    pool_y: list = []
    pool_cd: list = []
    unselected = [cid for cid in curves if cid not in sel_ids]
    for cid in unselected:
        xs, ys = curves[cid]
        pool_x.extend(xs + [None])
        pool_y.extend(ys + [None])
        pool_cd.extend([[ordinal[cid], cid]] * len(xs) + [[None, None]])
    if pool_x:
        fig.add_trace(go.Scattergl(
            x=pool_x, y=pool_y, mode="lines",
            line=dict(color="rgba(120,120,140,0.30)", width=1),
            customdata=pool_cd,
            hovertemplate=(f"conv #%{{customdata[0]}} — {x_hover}, "
                           "ctx %{y:,} tok<extra>click to select</extra>"),
            name="unselected",
        ))
    for cid in sorted(sel_ids, key=lambda c: ordinal.get(c, 0)):
        xs, ys = curves[cid]
        fig.add_trace(go.Scattergl(
            x=xs, y=ys, mode="lines",
            line=dict(color=color_for(cid), width=2.5),
            customdata=[[ordinal[cid], cid]] * len(xs),
            hovertemplate=(f"conv #%{{customdata[0]}} — {x_hover}, "
                           "ctx %{y:,} tok<extra>click to deselect</extra>"),
            name=f"#{ordinal.get(cid, '?')}",
        ))
    n_sel = len(sel_ids)
    band = theme.top_band_layout(
        f"Context growth — {len(curves)} conversations"
        + (f", {n_sel} selected" if n_sel else ""), title_size=14)
    band["showlegend"] = bool(sel_ids)
    fig.update_layout(**theme.base_layout(**band))
    fig.update_xaxes(title_text=CURVE_X_MEASURES[x_measure], title_font_size=12,
                     type=("log" if log_x else "linear"))
    fig.update_yaxes(type="log", title_text="context length (input tokens)",
                     title_font_size=12)
    return fig
