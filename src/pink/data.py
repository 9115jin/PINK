"""FERMAT (Nath et al., ACL 2025) as an OCR-faithfulness benchmark.

The dataset is downloaded from the Hugging Face Hub (``ai4bharat/FERMAT``, CC-BY-4.0).
It is gated: accept the terms on its page and log in (``huggingface-cli login``) first.

Each sample is a handwritten image of a math question and a student solution that
contains a (possibly superficial) error. The *oracle transcription* is the LaTeX of what
the student actually wrote (FERMAT's ``pert_a``), with FERMAT's ``\\textcolor{red}{...}``
markers around the inserted error removed so the judge cannot see where the error is.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

FERMAT_HF_ID = "ai4bharat/FERMAT"

_RED = "\\textcolor{red}{"


def _unwrap_red(text: str) -> str:
    """Replace every ``\\textcolor{red}{X}`` by ``X`` (brace-balanced, so nested LaTeX survives)."""
    out, i = [], 0
    while True:
        j = text.find(_RED, i)
        if j < 0:
            out.append(text[i:])
            return "".join(out)
        out.append(text[i:j])
        k, depth = j + len(_RED), 1
        while k < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[k], 0)
            k += 1
        out.append(text[j + len(_RED):k - 1] if depth == 0 else text[j + len(_RED):])
        i = k


def oracle_transcription(pert_a: str) -> str:
    """FERMAT ``pert_a`` -> oracle transcription used by PINK (red error markers removed)."""
    return _unwrap_red(str(pert_a)).replace("\r\n", "\n")


def _image_ext(data: bytes, path: Optional[str]) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if path and Path(path).suffix:
        return Path(path).suffix.lower()
    return ".png"


def load_fermat(split: str = "train", cache_dir: Optional[str] = None):
    """Load FERMAT as a ``datasets.Dataset`` with undecoded images (``{"bytes", "path"}``)."""
    try:
        from datasets import Image, load_dataset
    except ImportError as e:  # pragma: no cover
        raise ImportError("FERMAT support needs `pip install pink-eval[fermat]`") from e
    ds = load_dataset(FERMAT_HF_ID, split=split, cache_dir=cache_dir)
    return ds.cast_column("image", Image(decode=False))


def export_fermat(out_dir, limit: Optional[int] = None, ids: Optional[List[str]] = None,
                  cache_dir: Optional[str] = None, seed: int = 0, images: bool = True) -> Path:
    """Write ``images/`` and ``manifest.jsonl`` to ``out_dir``.

    Each manifest line holds ``sample_id``, ``image`` (relative path), ``problem`` (the
    question text given to the judge) and ``solution`` (the oracle transcription), plus a
    few FERMAT metadata fields. The manifest itself is the oracle input for ``pink grade``.
    With ``limit``, a random subset (fixed ``seed``) is exported instead of the full set.
    ``images=False`` writes only the manifest (enough for grading the oracle).
    """
    from ._runner import write_jsonl

    out_dir = Path(out_dir)
    if images:
        (out_dir / "images").mkdir(parents=True, exist_ok=True)
    ds = load_fermat(cache_dir=cache_dir)
    if ids:
        wanted = set(ids)
        ds = ds.filter(lambda r: r["new_custom_id"] in wanted)
    if limit is not None:
        ds = ds.shuffle(seed=seed).select(range(min(limit, len(ds))))

    rows: List[Dict[str, Any]] = []
    for r in ds:
        sid = r["new_custom_id"]
        img = r["image"]
        rel = f"images/{sid}{_image_ext(img['bytes'], img.get('path'))}"
        if images:
            (out_dir / rel).write_bytes(img["bytes"])
        rows.append({
            "sample_id": sid,
            "image": rel,
            "problem": str(r["orig_q"]).replace("\r\n", "\n"),
            "solution": oracle_transcription(r["pert_a"]),
            "domain": r.get("domain_code"),
            "grade": r.get("grade"),
            "has_error": r.get("has_error"),
        })
    manifest = out_dir / "manifest.jsonl"
    write_jsonl(manifest, rows)
    return manifest
