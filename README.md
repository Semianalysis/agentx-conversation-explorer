# AgentX Conversation Explorer

A standalone [Dash](https://dash.plotly.com/) app for exploring agentic-inference
trace datasets from [InferenceX AgentX](https://inferencex.semianalysis.com/agentx/)
— real Claude Code conversation traces, viewed through the lens of the serving
compute they imply.

## Quickstart

```bash
pip install -r requirements.txt
python app.py            # -> http://localhost:8050
```

The app opens on **Explorer** with nothing loaded. Its Dataset panel is the
working set — a list of the datasets this session has loaded, each with a tick
box and an **×** to unload it. Four buttons fill it:

- **Browse published…** — the Summary finder, where you search the published
  AgentX datasets, preview their stats and load one (use **Download traces** on
  its Summary card if its traces are not cached yet).
- **Local traces…** — pick an agent-trace file from this machine.
- **Import session…** / **Export session** — restore or save the working set
  (slugs, labels, where local traces came from, and which were ticked — never
  trace data), so resuming brings back the view you exported, not just the
  list. Importing names anything this machine no longer has rather than
  skipping it.

The first dataset you load is ticked for you and charts immediately; later ones
join the list for you to tick, so a load never silently changes what you are
looking at. **Tick more than one and they are explored together** — the
conversation list names each row's dataset, and conversation numbers stay
per-dataset, so two datasets can never collide. Unloading with × drops a
dataset from the session; the cached copy on disk is kept.

Traces are cached under `data/` (gitignored); every other tab reads only that
cache.

## Tabs

Every tab is a viewport onto the same state: the working set of datasets and
the conversation selection are shared everywhere; per-tab controls
(measures, scales, filters, inspectors) are local views. Hover the ⓘ tags anywhere
in the app for explanations of measures, columns, and controls.

- **Summary** (last tab) — summary cards for the datasets this session has
  loaded, plus a finder that searches published or locally imported datasets,
  previews their stats, and loads one into the session.
- **Explorer** — one context-growth curve per conversation (x: turn count /
  cumulative time / busy time), a serving-assumption bar (architecture, GPU,
  precisions, TP/PP/DP, MFU/MBU — all defaulting to documented values), and the
  sortable/filterable conversation list that drives the cross-tab selection.
- **Correlations** — three histograms with selectable measures (KV cache,
  uncached input, decode output, turn FLOPs) over turn/time/value bins. Click
  bars to build color-coded selections on one chart and see how much of every
  bar on the other charts correlates, stacked by color, with per-selection
  inspectors in the left panel.
- **Pause Analytics** — what happens in the dead air between turns: pause
  durations stacked by cause (subagent work from the trace structure, tool
  categories from an imported export, and an honest "unknown" remainder),
  alongside the ISL delivered when each pause ended, on one shared duration
  axis.
- **Deep-dive** — implied FLOPs / memory / network aggregated over the selected
  conversations, three scatter charts over a shared x measure, a click magnifier
  (5×) with cross-chart highlighting, and a per-request point inspector.

All compute/memory/network figures are **implied**: derived from the traces' token
counts under the serving assumptions you pick — nothing is measured on hardware,
and the app never fabricates missing values.

## Data

Per-conversation traces are fetched from the InferenceX AgentX API and cached
locally. The bundled dataset registry starts with `cc-traces-weka-062126`
(393 conversations, ~167k requests) and its 256k-context variant — both
published under Apache-2.0 on HuggingFace.

## Importing your own agent traces (OpenClaw exports)

Besides the AgentX datasets, the explorer can load an **OpenClaw trajectory
export** — the `steps.csv` / `tools.csv` / `runs.csv` bundle produced from
harness traces:

**From the app:** the Explorer tab's Dataset panel has **Local traces…** —
pick the file with the dialog and it is imported, added to the working set and
ticked, so it charts straight away.

**From the command line:**

```bash
python -m modules.openclaw_import path/to/export.zip      # or a directory
```

Imports stay on your machine: they are written to the local `data/` cache,
which is gitignored, so a private dataset cannot be committed or published by
accident. It writes the same local cache the AgentX importers use, so every tab
works on it unchanged. The mapping is
one conversation per agent lane and one turn per model call, with
`cacheRead` as cached tokens and `cacheWrite + input` as the uncached tokens
prefill actually computes. Pauses keep their timing (so the
"idle before turn" measure works), but tool calls, costs and compaction
counts are not represented — this app models turns, not tool execution.

## Internal mode (optional)

Operators who capture their own traces can point the explorer at additional
HuggingFace sources. Set `AGENTX_INTERNAL=1` to reveal an "Internal sources"
panel on the Summary tab; it lists the datasets of the namespaces in
`AGENTX_HF_SOURCES` (comma-separated, default `semianalysisai`) and imports
raw `cc-traces-weka`-format datasets locally. With a standard `HF_TOKEN` set,
private datasets your token can read appear too — access always follows
HuggingFace's own permissions. The default (public) configuration shows only
the datasets published on the AgentX site.

## License

Apache-2.0 — same license as the AgentX trace datasets this tool explores.
