---
name: agentx
description: Resume work on the AgentX Conversation Explorer. Orients on build / git / data-cache / test / server state, reports one compact status block, then continues the user's task under the project's standing rules (bump APP_BUILD on every change, ship via PR, user runs the server).
---

# Resume — AgentX Conversation Explorer

Run the orientation steps below (all promptless: Read/Glob/python/git), report ONE
compact status block, then continue with whatever the user asked — or, if they only
said "resume", ask nothing and summarize where the last session left off from memory.

## Orientation steps

1. **Build + code state**: Read `modules/version.py` (APP_BUILD). Run
   `git status -sb` and `git log --oneline -3` — note dirty files and ahead/behind
   vs `origin/main` (https://github.com/Semianalysis/agentx-data-explorer — the
   company org repo, canonical since 2026-08-29).
2. **Data cache**: count `data/*/conversations/*.json` per dataset (expect 393 each
   for cc-traces-weka-062126 and -256k). If empty: Overview tab → Import + Download
   traces refills it from the InferenceX API.
3. **Tests**: `python -m pytest tests/ -q` — if anything fails, fix before feature
   work.
4. **Server**: check port 8050 (`Get-NetTCPConnection -LocalPort 8050 -State Listen`).
   NEVER kill a listener that might be the user's own server — instead compare the
   served `<title>` build number against version.py and report drift ("browser shows
   bld N, disk is bld M — restart your server").

## Status block format

`bld N | git: clean/dirty, ahead X | data: 393+393 convs cached | tests: N passed | server: up (bld N) / down`

## Standing rules (enforce without being asked)

- **Bump `modules/version.py` APP_BUILD on EVERY app change** (commit message ends
  with the Claude co-author line).
- **Ship via PR — direct pushes to main are rejected** (org rules: PRs required +
  'Supply-chain cooldown check' workflow). Flow: commit on a `bld-N` branch, push,
  `gh pr create`, `gh pr merge --merge --auto`, wait for the merge, then sync main
  and delete the branch.
- The user runs `python app.py` themselves. Verify changes with a temporary
  background instance, then STOP it and confirm port 8050 is free.
- Wording: "Conversation Explorer" / "conversations" — never "Trace Explorer".
- Tabs are alternate views of ONE state: dataset (dropdowns on Explorer/Correlations
  are synced — one shared value), conversation selection, and serving assumptions
  are GLOBAL (shared stores; the two conversation lists are the same object).
  Everything else is a viewport option local to its tab (axis measures, log/linear
  scales, roles, models, correlation inspectors, sort/filter).
- Scale rule: ordinals (conv #, turn #) linear; magnitudes (time/tokens/bytes/FLOPs)
  log, clamped UP to 1 unit — never drop zeros.
- Never fabricate values; missing/undefined → None and drop, with gate counts shown.
