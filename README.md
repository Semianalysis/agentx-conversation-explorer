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

On first run, open the **Overview** tab, click **Import / refresh datasets**, then
**Download traces** on a dataset card. Traces are cached under `data/` (gitignored);
every other tab reads only that cache.

## Tabs

Every tab is a viewport onto the same state: the dataset, the conversation
selection, and the serving assumptions are shared everywhere; per-tab controls
(measures, scales, filters, inspectors) are local views. Hover the ⓘ tags anywhere
in the app for explanations of measures, columns, and controls.

- **Overview** — import datasets and see summary cards (token totals, model mix,
  cached fraction).
- **Explorer** — one context-growth curve per conversation (x: turn count /
  cumulative time / busy time), a serving-assumption bar (architecture, GPU,
  precisions, TP/PP/DP, MFU/MBU — all defaulting to documented values), and the
  sortable/filterable conversation list that drives the cross-tab selection.
- **Correlations** — three histograms with selectable measures (KV cache,
  uncached input, decode output, turn FLOPs) over turn/time/value bins. Click
  bars to build color-coded selections on one chart and see how much of every
  bar on the other charts correlates, stacked by color, with per-selection
  inspectors in the left panel.
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

## Internal mode (optional)

Operators who capture their own traces can point the explorer at additional
HuggingFace sources. Set `AGENTX_INTERNAL=1` to reveal an "Internal sources"
panel on the Overview tab; it lists the datasets of the namespaces in
`AGENTX_HF_SOURCES` (comma-separated, default `semianalysisai`) and imports
raw `cc-traces-weka`-format datasets locally. With a standard `HF_TOKEN` set,
private datasets your token can read appear too — access always follows
HuggingFace's own permissions. The default (public) configuration shows only
the datasets published on the AgentX site.

## License

Apache-2.0 — same license as the AgentX trace datasets this tool explores.
