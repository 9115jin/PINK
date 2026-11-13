"""PINK score: over-correction penalty (paper Eq. 2-3) and aggregation (Eq. 4).

For every rubric component r of a sample, with  dS_r = S_model,r - S_oracle,r :

    S'_r = S_model,r    if dS_r <= 0            (no over-correction)
           S_oracle,r   if 0 < dS_r <= T        (minor: revert to the oracle score)
           0            if dS_r > T             (major: nullify)

    S_penalized = sum_r S'_r
    PINK        = sum_n S_penalized,n / sum_n S_oracle,n

with T = 10 (half of a 20-point component).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Union

import numpy as np
import pandas as pd

from .rubric import RUBRIC_ITEMS, SCORE_COLUMNS

DEFAULT_THRESHOLD: float = 10.0

PathLike = Union[str, Path]


def penalize(model_scores, oracle_scores, threshold: float = DEFAULT_THRESHOLD) -> np.ndarray:
    """Apply the two-tier over-correction penalty element-wise (Eq. 2)."""
    m = np.asarray(model_scores, dtype=float)
    o = np.asarray(oracle_scores, dtype=float)
    delta = m - o
    return np.where(delta <= 0, m, np.where(delta <= threshold, o, 0.0))


def load_grades(path: PathLike) -> pd.DataFrame:
    """Load a grade file written by ``pink grade`` (CSV or JSONL).

    Returns successfully graded rows, indexed by ``sample_id``, with one column per
    rubric component (see :data:`pink.rubric.SCORE_COLUMNS`).
    """
    path = Path(path)
    if path.suffix == ".jsonl":
        df = pd.read_json(path, lines=True, dtype={"sample_id": str})
    else:
        df = pd.read_csv(path, dtype={"sample_id": str})
    missing = [c for c in ["sample_id", *SCORE_COLUMNS] if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    if "success" in df.columns:
        ok = df["success"].astype(str).str.lower().isin(["true", "1"])
        df = df[ok]
    df = df.dropna(subset=SCORE_COLUMNS)
    df = df.drop_duplicates("sample_id", keep="last").set_index("sample_id")
    return df[SCORE_COLUMNS].astype(float)


def pink_table(model: pd.DataFrame, oracle: pd.DataFrame, threshold: float = DEFAULT_THRESHOLD,
               missing: str = "drop") -> pd.DataFrame:
    """Per-sample table with model, oracle and penalized totals.

    ``model`` / ``oracle`` are DataFrames indexed by sample id with :data:`SCORE_COLUMNS`
    (e.g. from :func:`load_grades`). Only samples graded for the oracle are used.
    ``missing`` controls oracle samples without a model grade: ``"drop"`` excludes them
    (as in the paper), ``"zero"`` counts them as a 0-score transcription.
    """
    if missing not in ("drop", "zero"):
        raise ValueError("missing must be 'drop' or 'zero'")
    if missing == "drop":
        ids = oracle.index.intersection(model.index)
        m = model.loc[ids, SCORE_COLUMNS]
    else:
        ids = oracle.index
        m = model.reindex(ids)[SCORE_COLUMNS].fillna(0.0)
    o = oracle.loc[ids, SCORE_COLUMNS]

    out = pd.DataFrame(index=ids)
    out.index.name = "sample_id"
    pen = penalize(m.to_numpy(), o.to_numpy(), threshold)
    over = m.to_numpy() > o.to_numpy()
    for j in range(len(RUBRIC_ITEMS)):
        out[f"over_{SCORE_COLUMNS[j][len('score_'):]}"] = over[:, j]
    out["model_total"] = m.sum(axis=1).to_numpy()
    out["oracle_total"] = o.sum(axis=1).to_numpy()
    out["penalized_total"] = pen.sum(axis=1)
    out["over_corrected"] = over.any(axis=1)
    return out


@dataclass
class PINKResult:
    pink: float                    #: PINK score (Eq. 4)
    raw_ratio: float               #: sum(S_model) / sum(S_oracle), i.e. without the penalty
    over_correction_rate: float    #: share of samples with >= 1 over-corrected component
    n: int                         #: samples used
    n_oracle: int                  #: samples with an oracle grade
    n_model: int                   #: samples with a model grade
    threshold: float = DEFAULT_THRESHOLD
    over_corrections_per_item: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        lines = [
            f"PINK                 = {self.pink:.4f}",
            f"raw (no penalty)     = {self.raw_ratio:.4f}",
            f"over-correction rate = {self.over_correction_rate * 100:.1f}%",
            f"samples used         = {self.n}  (oracle graded: {self.n_oracle}, model graded: {self.n_model})",
        ]
        return "\n".join(lines)


def compute_pink(model: Union[pd.DataFrame, PathLike], oracle: Union[pd.DataFrame, PathLike],
                 threshold: float = DEFAULT_THRESHOLD, missing: str = "drop") -> PINKResult:
    """Compute PINK from model grades and oracle grades (DataFrames or grade-file paths)."""
    model_df = model if isinstance(model, pd.DataFrame) else load_grades(model)
    oracle_df = oracle if isinstance(oracle, pd.DataFrame) else load_grades(oracle)
    table = pink_table(model_df, oracle_df, threshold=threshold, missing=missing)

    oracle_sum = table["oracle_total"].sum()
    if len(table) == 0 or oracle_sum <= 0:
        pink = raw = float("nan")
    else:
        pink = float(table["penalized_total"].sum() / oracle_sum)
        raw = float(table["model_total"].sum() / oracle_sum)
    per_item = {
        item: int(table[f"over_{col[len('score_'):]}"].sum()) for item, col in zip(RUBRIC_ITEMS, SCORE_COLUMNS)
    }
    return PINKResult(
        pink=pink,
        raw_ratio=raw,
        over_correction_rate=float(table["over_corrected"].mean()) if len(table) else float("nan"),
        n=len(table),
        n_oracle=len(oracle_df),
        n_model=len(model_df),
        threshold=threshold,
        over_corrections_per_item=per_item,
    )


def result_to_json(result: PINKResult) -> str:
    return json.dumps(result.to_dict(), indent=2)
