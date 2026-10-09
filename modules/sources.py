"""Which datasets this session is working with — the SOURCE LIST.

Privacy stance: the app offers the published (website) datasets by default
and NEVER surfaces a locally imported dataset until the user brings it into
the session, either by importing an export or by loading it from the
Summary tab's finder. The source list is session state, exportable to a
small JSON file so the same working set can be picked up again tomorrow.

A source is {'slug', 'kind': 'website'|'local', 'label', 'path'|None}.
'path' records where a local export came from, so an exported session says
what to re-import; it is never used to read data behind the user's back.

Pure module: no Dash, no filesystem writes — the unit-test surface.
"""
from __future__ import annotations

from datetime import datetime, timezone

SCHEMA = "agentx-explorer/sources"
SCHEMA_VERSION = 1
WEBSITE, LOCAL = "website", "local"


def make_source(slug: str, kind: str, label: str,
                path: str | None = None) -> dict:
    """One source entry. Unknown kind -> ValueError (typo detection)."""
    if kind not in (WEBSITE, LOCAL):
        raise ValueError(f"unknown source kind {kind!r}; "
                         f"use {WEBSITE!r} or {LOCAL!r}")
    if not slug:
        raise ValueError("a source needs a slug")
    return {"slug": slug, "kind": kind, "label": label or slug, "path": path}


def source_kind(detail: dict) -> str:
    """Website datasets come from the AgentX API and carry no 'source' key;
    anything imported locally sets one (hf, openclaw)."""
    return LOCAL if detail.get("source") else WEBSITE


def add(sources: list[dict], source: dict) -> list[dict]:
    """Source list with `source` added or replaced (slug is the identity).
    Copy-on-write: the input list is never mutated."""
    out = [s for s in (sources or []) if s["slug"] != source["slug"]]
    out.append(source)
    return sorted(out, key=lambda s: (s["kind"] != WEBSITE, s["label"].lower()))


def remove(sources: list[dict], slug: str) -> list[dict]:
    return [s for s in (sources or []) if s["slug"] != slug]


def slugs(sources: list[dict], kind: str | None = None) -> list[str]:
    return [s["slug"] for s in (sources or [])
            if kind is None or s["kind"] == kind]


def export_blob(sources: list[dict], active: list[str] | None = None) -> dict:
    """The session file's contents. Carries only slugs, labels, the paths the
    user picked and which sources were TICKED — never trace data.

    'active' is what makes resuming restore the view and not merely the list.
    A slug that is not in `sources` raises: a session that ticks something it
    does not carry is a bug, not something to silently drop.
    """
    slugs = {s["slug"] for s in (sources or [])}
    ticked = [a for a in (active or [])]
    unknown = [a for a in ticked if a not in slugs]
    if unknown:
        raise ValueError(f"cannot export ticks for sources not in the "
                         f"session: {unknown[:3]}")
    return {
        "schema": SCHEMA, "version": SCHEMA_VERSION,
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": [make_source(s["slug"], s["kind"], s.get("label", ""),
                                s.get("path")) for s in (sources or [])],
        "active": ticked,
    }


def parse_blob(blob: dict) -> tuple[list[dict], list[str]]:
    """Session file -> (source list, ticked slugs). Raises with context on
    anything that is not one of ours: a mistyped file should say so, not
    silently load zero sources.

    A file written before sessions recorded their ticks has no 'active' key;
    every source it lists is treated as ticked, which is what those files
    meant.
    """
    if not isinstance(blob, dict):
        raise ValueError(f"session file is {type(blob).__name__}, not an object")
    if blob.get("schema") != SCHEMA:
        raise ValueError(f"not an explorer session file "
                         f"(schema={blob.get('schema')!r})")
    if blob.get("version") != SCHEMA_VERSION:
        raise ValueError(f"session file version {blob.get('version')!r} "
                         f"!= {SCHEMA_VERSION}")
    raw = blob.get("sources")
    if not isinstance(raw, list):
        raise ValueError("session file has no 'sources' list")
    sources = [make_source(s.get("slug", ""), s.get("kind", ""),
                           s.get("label", ""), s.get("path")) for s in raw]
    slugs = {s["slug"] for s in sources}
    ticked = blob.get("active")
    if ticked is None:
        return sources, [s["slug"] for s in sources]
    if not isinstance(ticked, list):
        raise ValueError(f"session file 'active' is "
                         f"{type(ticked).__name__}, not a list")
    unknown = [a for a in ticked if a not in slugs]
    if unknown:
        raise ValueError(f"session file ticks sources it does not list: "
                         f"{unknown[:3]}")
    return sources, list(ticked)


def resolve(sources: list[dict], available: set[str]) -> tuple[list[dict],
                                                               list[dict]]:
    """Split a restored source list into (present, missing) against the
    datasets actually cached on this machine. Missing ones are REPORTED,
    never silently dropped — a session that references an export this
    machine has not imported should say which."""
    present = [s for s in (sources or []) if s["slug"] in available]
    missing = [s for s in (sources or []) if s["slug"] not in available]
    return present, missing
