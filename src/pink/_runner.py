"""Parallel, resumable execution of per-sample API calls with incremental output."""

from __future__ import annotations

import csv
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import pandas as pd


def read_jsonl(path) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows: Iterable[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    if path.suffix == ".csv":
        return pd.read_csv(path, dtype={"sample_id": str}, keep_default_na=False).to_dict("records")
    return read_jsonl(path)


def _is_success(row: Dict[str, Any]) -> bool:
    return str(row.get("success", "")).lower() in ("true", "1")


def _compact(path: Path, columns: Optional[Sequence[str]]) -> None:
    """Keep one row per sample (the last successful one, else the last one)."""
    best: Dict[str, Dict[str, Any]] = {}
    for row in _read_rows(path):
        sid = str(row["sample_id"])
        if sid not in best or _is_success(row) or not _is_success(best[sid]):
            best[sid] = row
    rows = list(best.values())
    if path.suffix == ".csv":
        pd.DataFrame(rows, columns=list(columns) if columns else None).to_csv(path, index=False)
    else:
        write_jsonl(path, rows)


def run_jobs(items: Sequence[Dict[str, Any]], fn: Callable[[Dict[str, Any]], Dict[str, Any]], out_path,
             columns: Optional[Sequence[str]] = None, workers: int = 8, resume: bool = True,
             desc: str = "") -> Dict[str, int]:
    """Call ``fn(item)`` for every item not yet done and append each result row to ``out_path``.

    ``out_path`` is CSV (``columns`` required) or JSONL. Re-running the same command
    skips samples that already succeeded, so interrupted runs can simply be restarted.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not resume and out_path.exists():
        out_path.unlink()
    done = {str(r["sample_id"]) for r in _read_rows(out_path) if _is_success(r)}
    todo = [it for it in items if str(it["sample_id"]) not in done]

    lock = threading.Lock()
    is_csv = out_path.suffix == ".csv"
    new_file = not out_path.exists() or out_path.stat().st_size == 0
    fh = open(out_path, "a", encoding="utf-8", newline="")
    writer = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore") if is_csv else None
    if is_csv and new_file:
        writer.writeheader()

    try:
        from tqdm import tqdm
        bar = tqdm(total=len(todo), desc=desc or out_path.name)
    except ImportError:  # pragma: no cover
        bar = None

    n_ok = 0
    try:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = [pool.submit(fn, it) for it in todo]
            for fut in as_completed(futures):
                row = fut.result()
                with lock:
                    if is_csv:
                        writer.writerow(row)
                    else:
                        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fh.flush()
                n_ok += _is_success(row)
                if bar is not None:
                    bar.update(1)
    finally:
        fh.close()
        if bar is not None:
            bar.close()
        _compact(out_path, columns)
    return {"skipped": len(done), "attempted": len(todo), "succeeded": n_ok, "failed": len(todo) - n_ok}
