# Data source notes — cc-traces-weka-062126

State as of 2026-08-26. Target dataset page:
https://inferencex.semianalysis.com/agentx/cc-traces-weka-062126 (fully client-rendered
Next.js; no data in the HTML).

## Where the data actually lives

The site is open source: **github.com/SemiAnalysisAI/InferenceX-app** (monorepo,
`packages/app`). The dataset page component chain:

- `packages/app/src/app/agentx/[slug]/page.tsx` → `<DatasetDetail slug=...>`
- `packages/app/src/components/datasets/dataset-detail.tsx` → data via React Query hooks
  `useDataset(slug)` and `useDatasetConversations({slug, search, sort, limit, offset})`
  from `packages/app/src/hooks/api/use-datasets.ts`
- **NEXT STEP: fetch `use-datasets.ts` to get the exact REST endpoints and response
  types.** (Probed guesses `/api/agentx/*`, `/api/datasets/*` 404; the model_charts
  endpoints `/api/v1/availability` + `/api/v1/benchmarks?model=` are the GPU benchmark
  DB, a different dataset.)
- `dataset.hf_url` links to **HuggingFace** — the raw traces are published there.
  The site's `chart_data` is pre-binned histogram summaries; for our correlations and
  per-trace FLOP analysis we need the raw per-request records → download from HF
  (likely JSONL/parquet per conversation).

## Observed schema (from dataset-detail.tsx)

Dataset object: `label`, `variant`, `description`, `hf_url`, `conversation_count`,
`summary`, `chart_data`.

`summary` keys: `medianRequestsPerConversation`, `meanRequestsPerConversation`,
`mainTurns`, `medianSubagentsPerTrace`, `meanSubagentsPerTrace`, `cachedPct`,
`totalIn`, `totalOut`, `modelMix` ({model: turn_count}).

`chart_data` distributions (each feeds a DistributionCard, token dists log-scale):
`inputTokensPerTurn`, `outputTokensPerTurn`, `uncachedInputTokensPerTurn`,
`turnsPerConversation`, `subagentInputTokensPerRequest`,
`subagentOutputTokensPerRequest`, `cachedFractionPerTurn`.

Conversations listing (paged 50, sort tokens|turns|subagents|id, text search):
items = `{conv_id, num_turns, num_subagent_groups, total_in, total_out, total_cached,
models[]}` + `total`.

Per-conversation view exists (`/agentx/[slug]/conversations/[convId]`,
`conversation-view.tsx`, `trace-flamegraph.tsx` + `trace-flamegraph-model.ts`) — implies
a per-conversation API with per-request timing/token detail. Related repo:
**github.com/SemiAnalysisAI/agentx-harness** (the trace producer — check it for the raw
trace record format).

## Mapping to our app's vocabulary

- "turn"/request record ≈ one API call: cached (context) input tokens + uncached (new)
  input tokens + output tokens. `context ≈ total_in per request`, `new input ≈ uncached
  input`, `decode/output ≈ output tokens`.
- GPU type is NOT in the trace dataset (traces are client-side agent recordings); GPU
  enters via the deep-dive tab where we pair a trace with a model architecture + GPU
  config to compute FLOPs / memory / network. "Subsets by model" comes from the trace's
  `models[]` / per-request model field. (If a GPU-labeled variant of the dataset shows
  up in the API, revisit.)

## Resolved 2026-08-27 (all four questions answered)

1. **Endpoints** (no auth, from `use-datasets.ts`), base `https://inferencex.semianalysis.com/api/v1`:
   - `GET /datasets` → registry (2 datasets: `cc-traces-weka-062126` + `-256k`, 393 convs each)
   - `GET /datasets/{slug}` → detail incl. `summary` + pre-binned `chart_data` (v3)
   - `GET /datasets/{slug}/conversations?limit&offset&sort&search` → `{items, total}` (paged)
   - `GET /datasets/{slug}/conversations/{convId}` → `{..., structure: {blockSize, totals, nodes}}`
2. **No HF download needed.** `structure.nodes` IS the per-request record set: tree of
   `{kind: "turn"|"subagent", model, in, cached, uncached, out, startS, endS, turnIndex,
   rawIndex}`; subagent nodes carry `agentId` + `children` (recursive) and aggregate
   token sums (children are the ground truth — don't double-count). Invariant
   `in == cached + uncached` verified to hold on all ~167k records of both datasets.
   HF raw traces (`hash_ids` per request, 64-token cache blocks) only needed if we ever
   want block-level cache-hit simulation; format documented in
   agentx-harness `docs/tutorials/weka-trace.md`.
3. **Yes — per-request timestamps**: `startS`/`endS` seconds from conversation start.
4. **No GPU labels anywhere in the API** — confirmed; hardware enters only via the
   deep-dive pairing (trace × assumed architecture × GPU preset).

The app caches everything under `data/` (gitignored): `datasets.json`,
`<slug>/detail.json`, `<slug>/conversations_index.json`, `<slug>/conversations/<id>.json`.
