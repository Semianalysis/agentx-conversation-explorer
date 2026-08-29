# AgentX Data Explorer

A standalone [Dash](https://dash.plotly.com/) app for exploring the **AgentX
agentic-inference trace datasets** published by SemiAnalysis at
[inferencex.semianalysis.com/agentx](https://inferencex.semianalysis.com/agentx/) —
real multi-turn Claude Code coding sessions (main agent + subagents, per-request
token and timing detail) — and for estimating the compute, memory-bandwidth, and
network activity those traces imply on an assumed serving stack.

## Quick start

```bash
pip install -r requirements.txt
python app.py            # -> http://localhost:8050
```

On first run, open the **Overview** tab, click *Import / refresh datasets*, then
*Download traces* for each dataset (fetched once from the public InferenceX API
into a local `data/` cache, ~40 MB per dataset).

```bash
python -m pytest tests/  # unit tests
```

## Tabs

- **Overview** — dataset registry cards: token totals, cached fraction, model mix.
- **Explorer** — one context-growth curve per conversation (turn count vs context
  length); a sortable, range-filterable conversation list (shift-click checkbox
  ranges) that drives a cross-tab selection; serving-assumption controls (all
  defaulting to "any").
- **Turns** — per-request histograms (context, new uncached input, decode output),
  filterable by model and main/subagent role.
- **Correlations** — condition on a range of one dimension (box-select or min/max),
  see conditional histograms of the others.
- **Deep-dive** — aggregate implied FLOPs / HBM movement / interconnect traffic /
  single-GPU-equivalent time over the selected conversations, with a measure matrix
  (one shared x, three y axes) and grouped mean / min-max envelope modes.

## Data source

The app consumes the InferenceX public REST API (`/api/v1/datasets/...`); the
per-conversation `structure` endpoint carries per-request records
(`model, in, cached, uncached, out, startS, endS`) with the invariant
`in == cached + uncached`. Raw traces are also mirrored on
[HuggingFace](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126).
See `docs/data-source-notes.md`.

## A note on the estimates

Model internals for the traced models are not public. The deep-dive pairs observed
token/timing data with a **user-chosen architecture preset** (dense or MoE at
several scales) and GPU preset; all FLOPs/bytes/network figures are standard
roofline estimates — formulas documented in `modules/arch.py` — not measurements.

## Layout

```
app.py                     Dash assembly (tabs, build stamp)
modules/<tab>_layout.py    layout + that tab's dcc.Stores
modules/<tab>_callbacks.py callbacks (registered per tab)
modules/*_data.py, arch.py, measures.py, records.py   pure data/math (unit-test surface)
assets/*.js                shift-click range select, ctrl-click sort semantics
tests/                     pytest suite
```
