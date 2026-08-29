"""InferenceX AgentX API client with a local disk cache.

Endpoints (from InferenceX-app packages/app/src/hooks/api/use-datasets.ts):
  GET /api/v1/datasets                                    -> DatasetRecord[]
  GET /api/v1/datasets/{slug}                             -> DatasetDetail (summary + chart_data)
  GET /api/v1/datasets/{slug}/conversations?limit&offset  -> {items, total}
  GET /api/v1/datasets/{slug}/conversations/{convId}      -> ConversationDetail (structure.nodes)

Pure module: no Dash imports. All network fetches follow the fetcher discipline:
pre-defaulted results, bounded TOTAL wall-clock via as_completed(timeout=...),
shutdown(wait=False, cancel_futures=True).
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import logging
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

API_BASE = "https://inferencex.semianalysis.com/api/v1"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

_REQUEST_TIMEOUT_S = 60
_PAGE_LIMIT = 100


def _get_json(url: str) -> dict | list:
    resp = requests.get(url, timeout=_REQUEST_TIMEOUT_S)
    resp.raise_for_status()
    return resp.json()


def _cache_path(*parts: str) -> Path:
    p = DATA_DIR.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _read_cache(path: Path):
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_cache(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    tmp.replace(path)


# --- Registry + detail ---

def fetch_datasets(refresh: bool = False) -> list[dict]:
    """All ingested datasets (registry cards). Cached at data/datasets.json."""
    path = _cache_path("datasets.json")
    if not refresh:
        cached = _read_cache(path)
        if cached is not None:
            return cached
    data = _get_json(f"{API_BASE}/datasets")
    if not isinstance(data, list) or not data:
        raise ValueError(f"datasets endpoint returned {type(data).__name__} with no entries")
    _write_cache(path, data)
    return data


def fetch_dataset_detail(slug: str, refresh: bool = False) -> dict:
    """One dataset incl. summary + pre-binned chart_data."""
    path = _cache_path(slug, "detail.json")
    if not refresh:
        cached = _read_cache(path)
        if cached is not None:
            return cached
    data = _get_json(f"{API_BASE}/datasets/{slug}")
    if not isinstance(data, dict) or "summary" not in data:
        raise ValueError(f"dataset detail for {slug!r} missing 'summary': keys={sorted(data) if isinstance(data, dict) else data}")
    _write_cache(path, data)
    return data


def fetch_conversation_index(slug: str, refresh: bool = False) -> list[dict]:
    """Full conversation listing (counts only), paged. Cached per dataset."""
    path = _cache_path(slug, "conversations_index.json")
    if not refresh:
        cached = _read_cache(path)
        if cached is not None:
            return cached
    items: list[dict] = []
    offset = 0
    total = None
    while True:
        page = _get_json(
            f"{API_BASE}/datasets/{slug}/conversations?limit={_PAGE_LIMIT}&offset={offset}&sort=tokens"
        )
        page_items = page["items"]
        total = page["total"]
        items.extend(page_items)
        offset += len(page_items)
        if not page_items or offset >= total:
            break
    if total is not None and len(items) != total:
        raise ValueError(f"conversation index for {slug!r}: paged {len(items)} items but total={total}")
    _write_cache(path, items)
    return items


# --- Per-conversation detail cache ---

def conversation_dir(slug: str) -> Path:
    return DATA_DIR / slug / "conversations"


def cached_conversation_ids(slug: str) -> set[str]:
    d = conversation_dir(slug)
    if not d.is_dir():
        return set()
    return {p.stem for p in d.glob("*.json")}


def fetch_conversation_detail(slug: str, conv_id: str, refresh: bool = False) -> dict:
    path = _cache_path(slug, "conversations", f"{conv_id}.json")
    if not refresh:
        cached = _read_cache(path)
        if cached is not None:
            return cached
    data = _get_json(f"{API_BASE}/datasets/{slug}/conversations/{conv_id}")
    if not isinstance(data, dict) or "structure" not in data:
        raise ValueError(f"conversation {conv_id!r} of {slug!r} has no 'structure'")
    _write_cache(path, data)
    return data


def bulk_download_conversations(
    slug: str,
    max_workers: int = 12,
    total_timeout_s: float = 900.0,
) -> dict:
    """Download every missing conversation detail for `slug` into the disk cache.

    Returns {'requested', 'ok', 'failed', 'already_cached', 'elapsed_s', 'errors'}.
    Never raises on individual fetch failures; failures are counted and sampled
    (the caller reports them — a partial cache is usable, not fabricated).
    """
    index = fetch_conversation_index(slug)
    have = cached_conversation_ids(slug)
    missing = [it["conv_id"] for it in index if it["conv_id"] not in have]
    result = {
        "requested": len(missing),
        "ok": 0,
        "failed": 0,
        "already_cached": len(have),
        "elapsed_s": 0.0,
        "errors": [],
    }
    if not missing:
        return result

    t0 = time.monotonic()
    pool = cf.ThreadPoolExecutor(max_workers=max_workers)
    try:
        futures = {
            pool.submit(fetch_conversation_detail, slug, cid): cid for cid in missing
        }
        for fut in cf.as_completed(futures, timeout=total_timeout_s):
            cid = futures[fut]
            try:
                fut.result()
                result["ok"] += 1
            except Exception as exc:  # noqa: BLE001 - counted, sampled, reported
                result["failed"] += 1
                if len(result["errors"]) < 5:
                    result["errors"].append(f"{cid}: {exc}")
    except cf.TimeoutError:
        undone = result["requested"] - result["ok"] - result["failed"]
        result["failed"] += undone
        result["errors"].append(f"total wall-clock {total_timeout_s}s exceeded; {undone} fetches abandoned")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    result["elapsed_s"] = time.monotonic() - t0
    logger.info("bulk_download %s: %s", slug, result)
    return result
