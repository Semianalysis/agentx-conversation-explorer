# InferenceX Explorer — patterns to imitate

Condensed 2026-08-26 from a full survey of
`Business-Helper/app/dashboards/model_charts` (`modules/ix_explorer_*.py`,
`assets/ix_explorer.js`, `dashboard.py`). This is the interaction/layout model for the
agent-traces app. The source app runs Dash 3.x (`dash==3.1.1`,
`dash-bootstrap-components`, `plotly==6.2.0`) mounted on Flask; ours is standalone.

## Layout (the part the user likes)

Root div = column flex, `height: calc(100vh - 60px)`, `overflow: hidden`, four bands:

1. **Top bar** (flex row): primary entity dropdown (`multi=True`) + status text + action button.
2. **Import bar**: source inputs + import button + status div.
3. **Focus badge row**: clickable badges for quick focus filtering.
4. **3-column main area** (flex row, `flex: 1 1 auto`, `minHeight: 0`):
   - LEFT sidebar `flex: 0 0 260px`, `overflowY: auto` — all filters, axis pickers, options.
   - CENTER `flex: 1 1 0%`, `minWidth: 0` — one `dcc.Graph` filling the space,
     config `{displaylogo: False, modeBarButtonsToAdd: ["lasso2d","select2d"], scrollZoom: True}`.
   - RIGHT `flex: 0 0 640px` — custom legend panel (top, `maxHeight: 33vh`, monospace 13px)
     + selection/detail cards (bottom). Plotly's built-in legend is OFF.

Styling: inline `style=` dicts + flexbox; Bootstrap only for `dbc.Button`/`dbc.Modal`.
Layout constants at module top: `_F_SMALL="13px"`, `_F_BASE="15px"`, `_LEFT_W="260px"`.
Hard-won flexbox rule: every flex child that must not be crushed carries `minWidth: 0`
(+ explicit `maxWidth`); panels get defensive `maxHeight` in `vh`.

Left-sidebar control factories keep it DRY: `_filter_dropdown(id_, placeholder)`,
`_axis_dropdown(id_, placeholder)`, `_label(text)`.

## State architecture

- ALL of a tab's `dcc.Store`s live in that tab's layout module (components must exist
  before callbacks reference them).
- **Coalescer stores**: small callbacks mirror widget groups into one store
  (11 filter widgets → `filter-store`; axis dropdowns → `axis-store`), and the render
  callback takes the coalesced stores as Inputs.
- **Pool as State**: the full row pool (`rows-store`) is passed as `State` to render
  callbacks so it never re-transmits; the intermediate `filtered-rows-store` holds ONLY
  uids.
- Orthogonal visual states get separate stores: `hidden-series`, `focused-model` /
  `focused-series` (mutually exclusive pair reconciled in one callback), `selection`.
- Load-once pattern: `auto-loaded-store` boolean guards the bulk load; refresh = reload
  page or explicit Clear-pool button.

## Series identity (steal this)

A "series" = one config varying only along the sweep axis (there: concurrency; here:
e.g. one conversation, or one model+gpu group). Identity = canonical tuple of the
identity fields → `sha1(json.dumps(identity, sort_keys=True))[:16]` = `series_id`.
Color = `PALETTE[hash % n]`, marker = `MARKERS[hash % m]` — **hash-derived, not
index-derived**, so colors are stable across re-renders and filter changes.
Rows are stamped once at load: `_uid`, `_series_id`, `_series_label`,
`_series_label_compact`, `_color`, `_marker_symbol`.

## Interaction details

- Click any point → select its whole series; `customdata` on every point carries
  `[uid, series_id, ...]`; lasso `selectedData` preferred over `clickData`.
- Pattern-matching IDs for interactive lists:
  `{"type": "at-<thing>", "series_id": sid}`. Guard idiom in every pattern callback:
  `triggered_id` must be a dict, and
  `any(item.get("value") for item in callback_context.triggered)` must be truthy
  (re-render resets all `n_clicks=0` and must not fire the callback).
- NEVER render the same pattern-matched ID twice in the DOM (e.g. panel + modal) —
  Dash sees two stale-but-equal n_clicks and the callback stops firing.
- Empty-chart diagnostics: count rows through each gate
  (`n_pool → n_uid_match → n_visible → n_xy_ok`) and, when zero traces form, render a
  Plotly annotation naming which gate ate the rows + per-group row counts.
- Hover convention: `hovermode="closest"`, single tooltip beside the point,
  `bgcolor="rgba(255,255,255,0.75)"`, dark border, black monospace, left-aligned.
- Range-subset UX for correlations: Plotly `select2d`/`lasso2d` box select on the source
  histogram (or min/max numeric inputs) → store `{dim, lo, hi}` as view intent → other
  histograms re-render conditioned on the selection.

## Server/data conventions

- Server callbacks touching data are FEW (bulk load, import, clear). Everything
  derivable from loaded rows is a view computation.
- Fetcher discipline (learned from a production outage): default every result before
  fetching, bound TOTAL wall-clock with `as_completed(futures, timeout=...)`, and
  `pool.shutdown(wait=False, cancel_futures=True)` in `finally` — per-request timeouts
  do NOT cover DNS.
- Import status strings are explicit: "Loaded 12,345 rows from 7 models in 1.3s".

## Test patterns

- Pure `*_data.py` helpers: exhaustive unit tests, including raise-on-malformed.
- Record shape: a contract test asserting required keys/types on real fixture rows.
- Callbacks: test the module-level helper functions, not simulated Dash.
- Two pytest tiers via markers if the suite grows (fast default, `-m ""` for deep).
