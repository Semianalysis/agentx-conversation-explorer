# agent-traces — AgentX Conversation Explorer

(App title: "AgentX Conversation Explorer"; GitHub repo name: "AgentX Data Explorer" /
`agentx-data-explorer`. The user renamed it from "Trace Explorer" 2026-08-29 —
prefer "conversation" over "trace" in user-facing wording.)

Standalone Dash app for exploring agentic-inference trace datasets from
https://inferencex.semianalysis.com/agentx/ (starting with `cc-traces-weka-062126`).
Independent project — it imitates the InferenceX Explorer tab of
`Business-Helper/app/dashboards/model_charts` in style and interaction, but shares no code.

## Assignment (standing spec)

- **One state, many viewports (standing principle, restated by the user 2026-08-29).**
  Dataset, conversation selection, and serving assumptions are GLOBAL — switching
  tabs retains them. The dataset dropdowns on Explorer/Correlations are synced views
  of ONE value (sync_updates + self-loop callback in explorer_callbacks); the
  conversation selection store and assumption config flow from Explorer everywhere.
  Everything else a tab offers is a VIEWPORT option local to that tab: axis/measure
  choices, log/linear scales, role and model filters, correlation
  inspectors/selections, list sort/filter.
- **Overview tab** — import trace data, summarize the datasets found (counts, token totals,
  model mix, cached fraction). When an Explorer selection exists, a selection-scoped
  summary card renders above the dataset cards.
- **Explorer tab** (added 2026-08-27) — the main working view: one growth curve per
  conversation (y = context tokens log; x measure radio (bld 21): turn count |
  cumulative time | busy time — time values clamp UP to 1 s — plus the x
  linear/log scale radio). Serving-assumption
  dropdowns (arch, GPU, Wq/KVq, TP/PP/DP, MFU/MBU) sit in a TOP BAR, ALL defaulting
  "any" (= DEFAULT_ASSUMPTIONS in arch.py); left column holds only dataset + selection.
  The conversation list (identity = ordinal from the dataset's token-sorted index) is
  the filtering surface: native multi-sort (header clicks stack) + native filter row
  for ranges; columns = model, turns, final ctx, initial ISL, hours, resolved
  GPU/Wq/KVq/TP/PP/DP, and computed sustained prefill/decode GPU counts
  (implied GPU-seconds ÷ wall-clock, 3 sig figs, None when wall-clock is 0).
  List filtering scopes the chart (via `at-explorer-visible-store`). Row checkboxes,
  cell-drag ranges, and curve clicks all set the cross-tab selection store
  (`at-explorer-selection-store`), which scopes Overview/Turns/Correlations and
  Deep-dive's list.
- **Turns tab** — DROPPED at bld 19 (user 2026-08-29: Correlations covers it). Its
  pure binning helpers live on as modules/binning.py.
- **Correlations tab** (redesigned bld 17/19, measures + cursor model bld 20) —
  three histogram charts (slots y1/y2/y3) with a deep-dive-style measure matrix:
  x mode ∈ {token count, turn #, cumulative time, busy time} (+ log/linear bins
  radio), y measures ∈ {KV cache size (y1 default), new input (y2), decode output
  (y3), turn FLOPs}. token-count mode = each chart bins its OWN y measure, bar
  height = request count; positional modes = all charts share x bins, bar height =
  per-bin SUM of each chart's measure. Measures read the ENRICHED rows from
  deepdive_data.aggregate_selection (memoized in correlations_callbacks), so KV
  bytes / FLOPs follow the global serving assumptions. SELECTIONS: color-coded
  inspectors partition ONE shared selection chart (first pick anchors it); bins are
  EXCLUSIVE — picking with another inspector's armed cursor transfers a bin, picking
  an owned bin releases it; all inspectors empty → unanchored, first pick may land
  anywhere. Click an inspector's ➤ arrow to ARM it: chart wrappers get a
  corr-cursor-<k> class (assets/corr_cursors.css, generated from PALETTE) that
  recolors the mouse cursor over the selection chart (over all charts while
  unanchored); non-selection charts keep the default cursor and IGNORE clicks.
  Conditioning ALWAYS applies (no apply button): each non-empty selection's
  contribution stacks on the other charts in its color (member counts, or member
  measure sums in positional mode) — section size = how much of that bar correlates.
  Full pool stays a gray step silhouette; edges always from the full pool (never
  re-range). Inspector: colored Clear (section stays; never disappears, ≥1 always),
  clickable all-bins strip (other owners' bins shown faded in THEIR color; clicking
  transfers), per-run detail + Σ contribution per other chart. "Add selection" below
  the sections. Status line: "selections are in <selection-chart measure>" / "you
  may choose to begin a selection in any chart". Selections RESET when dataset /
  models / roles / measures / x-scale / assumptions / Explorer selection change.
  State machine pure in correlations_data.py; charts flex-fill the window height.
- **Trace deep-dive tab** — implied FLOPs / memory / network aggregated over the
  CHECKED conversations (nothing checked = all). The conversation list is the SAME
  OBJECT as the Explorer list: one data callback feeds both tables identical rows, one
  selection callback syncs checkboxes both ways (sort/filter stays viewport-local) —
  by construction they can never diverge. Shift-click on a checkbox toggles a whole
  displayed range (assets/shift_select.js replays real clicks through React). Charts:
  three stacked graphs sharing ONE x measure with independent y1/y2/y3, picked in a
  radio matrix (measures down the left, columns x|y1|y2|y3 — see modules/measures.py).
  Scale rule: ordinals (conv #, turn #) linear; every magnitude (time/tokens/bytes/
  FLOPs) log with values clamped UP to 1 unit so zeros/sub-unit values stay visible at
  the axis floor. An x-scale radio (auto/linear/log, bld 17) can override the x
  measure's natural scale; the three charts share one range-locked x axis either way. Dataset + assumptions come from Explorer ("any" → DEFAULT_ASSUMPTIONS
  in arch.py).
- Purpose: explore compute activity and configuration potential implied by real agent traces.

Data-source findings and open questions: `docs/data-source-notes.md`.
Interaction/layout patterns to imitate: `docs/ix-explorer-reference.md`.

## Coding style — the minimal set

Distilled 2026-08-26 from model_charts guidance (`CLAUDE.md` Engineering Method,
`docs/ai_guidance/mc-code-style-core.md`, `mc-dash-state-ux.md`, rulings audit). These are
the load-bearing rules; everything else is ordinary Python taste.

### Fail fast, loudly, at the right altitude

- Computation and data code lets exceptions propagate. Never return `0`, `None`, or `[]`
  in place of a failed computation — **never fabricate**. Missing optional data → `None`
  so the caller skips the point; malformed or impossible input → raise with context.
- UI callbacks get exactly ONE outer `try/except` → `logger.exception(...)` +
  `PreventUpdate` (or a visible error message). Banned everywhere: `except: pass`,
  `except: return <default>`. Silent defaults produce plausible-but-wrong charts.
- Validate at boundaries, loudly: unknown filter key → `KeyError` (typo detection);
  non-empty rows yielding zero usable metrics → `ValueError` naming what the user would
  otherwise stare at.
- Diagnostics over blank output: when a chart renders empty, annotate WHICH gate ate the
  rows (`n_pool → n_filtered → n_visible → n_xy_ok`), never show a silent empty plot.

### Assertions & contracts

- State the invariant a change must preserve as pre/postcondition assertions at the
  touched seams; design so correctness follows by induction.
- Name units in variable names whenever ambiguity is possible: `_bytes`, `_gb`, `_tokens`,
  `_ms`, `_s`, `_flops`, `_gpu`. Convert units at ONE point (e.g. seconds→ms at axis
  extraction), never scattered.
- Transient, build-tagged debug traces at risky seams are encouraged — and removed once
  the seam is proven.

### Brave coding

- Rapid progress over perceived risk: don't shrink or defer a change because adding code
  feels risky — instrument the risk (assertions, tripwires) and ship.
- Deliver a feature in ONE complete pass (data + UI + tests). Do not stage delivery into
  separately-gated increments; a half-migrated state is what hides integration bugs.
- When a format changes, migrate every producer and parser in the same pass, and delete
  the false/legacy paths in the same sweep.

### Reliable style

- One source of truth. Registries over callsite defaults; extract shared code on the
  third repeat.
- Identifiers over strings: retrieval by uid / stable hash key; display labels are
  projections of data, never lookup keys. (e.g. series identity = hash of the canonical
  config tuple → stable colors/markers across renders.)
- Small deterministic pure helpers with typed boundaries. Pure data modules import zero
  Dash — they are the unit-test surface.
- Module split per tab: `<tab>_layout.py` (layout + ALL of that tab's `dcc.Store`s),
  `<tab>_callbacks.py` (one `register_<tab>_callbacks(app)`; testable helpers at module
  level), `<tab>_data.py` (pure algorithms).
- Loaded records are immutable archives; derive, don't mutate. Copy-on-write when
  building saved/exported records.
- Every network fetch: pre-default results, bound TOTAL wall-clock
  (`as_completed(..., timeout=...)`), `shutdown(wait=False, cancel_futures=True)` —
  never let a stuck fetch hold a worker.
- Tests in `tests/test_<feature>.py`, pytest + unittest compatible. Test invariants and
  failure paths (malformed rows raise; empty pools annotate), not just happy paths.
  Contract-test the record shape.

### Dash specifics (this app)

- Single-user local app: plain `Dash(__name__)` + `app.run(debug=...)`; server-side
  callbacks in Python are fine (no clientside-JS mirroring needed at this scale) — keep
  the pure-data / callback split anyway.
- Component IDs: kebab-case `at-<tab>-<concern>[-<sub>]`; stores `at-<tab>-<name>-store`;
  buttons end `-btn`. `"-none-"` is the null sentinel for all dropdowns.
- Mirror widget groups into a single coalescer `dcc.Store` (filters → filter-store),
  and let downstream callbacks take the store as the Input. Pass big row pools as
  `State`, never `Input`; intermediate stores carry uids, not rows.
- `prevent_initial_call=True` on every callback that needs user data.
- Plotly hover: `hovermode="closest"`, translucent white tooltip
  (`rgba(255,255,255,0.75)`, dark border, monospace, left-aligned) beside the point.
- Histograms of token lengths default to log-x; keep linear toggle.

## Build number

`modules/version.py` `APP_BUILD` is displayed in the header ("AgentX Trace Explorer
bld N"). BUMP IT on every app change, no exceptions — the user uses it to tell a stale
browser page from the running server. Windows note: killing the server's wrapper shell
can orphan python.exe while a new instance also binds 8050 (no exclusive bind) —
requests then hit either server randomly. Before restarting, verify the port owner
(`Get-NetTCPConnection -LocalPort 8050`) and kill stray pythons.

## Commands

```bash
python -m pytest tests/          # unit tests
python app.py                    # run the app (http://localhost:8050)
```

## Autonomy

`.claude/settings.json` carries the standing permission allowlist (including read access
to `Business-Helper`). Work autonomously; batch questions; never park durable rules in
`settings.local.json` (the harness overwrites it).
