"""LLM judge: rubric-based semantic grading (paper Sec. 3.1).

Any OpenAI-compatible chat-completions endpoint works: the OpenAI API (GPT-5, the
paper's main judge) or a self-hosted model served with vLLM (e.g.
Qwen3-Next-80B-A3B-Thinking, the paper's open-source judge, see App. E).
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

from .rubric import (
    GPT5_PAPER_INSTRUCTIONS,
    GRADING_INSTRUCTIONS,
    JUSTIFICATION_COLUMNS,
    POINTS_PER_ITEM,
    RUBRIC_ITEMS,
    SCORE_COLUMNS,
    build_user_message,
    canonical_item,
)


@dataclass
class JudgeConfig:
    """How to reach the judge and which sampling parameters to send.

    Parameters left as ``None`` are not sent, so the endpoint's defaults apply.
    """

    model: str
    base_url: Optional[str] = None          #: ``None`` = OpenAI API
    api_key: Optional[str] = None           #: default: $OPENAI_API_KEY, or "EMPTY" for local servers
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    reasoning_effort: Optional[str] = None  #: OpenAI reasoning models only
    extra_body: Dict[str, Any] = field(default_factory=dict)
    instructions: str = GRADING_INSTRUCTIONS  #: system prompt (rubric) sent to the judge
    max_attempts: int = 5                   #: retries when the response cannot be parsed
    timeout: float = 600.0


#: Judges used in the paper. ``gpt-5`` is the main judge and uses the exact instructions
#: of the paper's GPT-5 runs, so its grades match the bundled ``fermat:gpt-5`` oracle
#: grades and the paper's leaderboard. ``qwen3-next-80b`` is the open-source judge
#: (r=0.99 / Kendall tau=0.90 with GPT-5 at the VLM level, App. E.1) with the Fig. 16 prompt.
JUDGE_PRESETS: Dict[str, JudgeConfig] = {
    "gpt-5": JudgeConfig(model="gpt-5-2025-08-07", reasoning_effort="medium",
                         instructions=GPT5_PAPER_INSTRUCTIONS),
    "qwen3-next-80b": JudgeConfig(
        model="Qwen/Qwen3-Next-80B-A3B-Thinking",
        base_url="http://localhost:8000/v1",
        temperature=0.6,
        top_p=0.95,
    ),
}


def judge_from_preset(name: str, **overrides: Any) -> JudgeConfig:
    if name not in JUDGE_PRESETS:
        raise KeyError(f"unknown judge preset {name!r}; choose from {sorted(JUDGE_PRESETS)}")
    overrides = {k: v for k, v in overrides.items() if v is not None}
    return replace(JUDGE_PRESETS[name], **overrides)


def _is_openai(cfg: JudgeConfig) -> bool:
    return cfg.base_url is None or "api.openai.com" in cfg.base_url


def make_client(cfg: JudgeConfig):
    from openai import OpenAI

    key = cfg.api_key or os.environ.get("OPENAI_API_KEY") or ("EMPTY" if not _is_openai(cfg) else None)
    return OpenAI(api_key=key, base_url=cfg.base_url, timeout=cfg.timeout)


# --------------------------------------------------------------------------- parsing

def parse_judge_response(text: str) -> Dict[str, Any]:
    """Extract the JSON grading object from a (possibly noisy) judge response."""
    if text is None:
        raise ValueError("empty response")
    text = text.strip()
    if "</think>" in text:  # reasoning trace without an opening tag
        text = text.rsplit("</think>", 1)[1].strip()
    text = re.sub(r"<(think|thought|reasoning)>[\s\S]*?</\1>", "", text).strip()

    fences = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    for block in reversed(fences):
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            continue
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    candidates: List[str] = []
    depth, start = 0, -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0:
                candidates.append(text[start:i + 1])
    pool = [c for c in candidates if '"scores"' in c] or candidates
    for cand in reversed(pool):
        for attempt in (cand, re.sub(r",\s*([}\]])", r"\1", cand)):
            try:
                return json.loads(attempt)
            except json.JSONDecodeError:
                continue
    raise ValueError(f"could not parse JSON from judge response: {text[:200]!r}")


def _to_score(value: Any) -> int:
    if isinstance(value, str) and "/" in value:
        value = value.split("/")[0]
    score = int(round(float(value)))
    return max(0, min(POINTS_PER_ITEM, score))


def extract_scores(parsed: Dict[str, Any]) -> Tuple[Dict[str, int], Dict[str, str]]:
    """Return ``(scores, justifications)`` keyed by rubric item; raises if an item is missing."""
    entries = parsed.get("scores") if isinstance(parsed, dict) else None
    if not isinstance(entries, list):
        raise ValueError("judge response has no 'scores' list")
    entries = [e for e in entries if isinstance(e, dict)]

    scores: Dict[str, int] = {}
    justifs: Dict[str, str] = {}
    for e in entries:
        item = canonical_item(str(e.get("item", "")))
        if item is not None and item not in scores:
            scores[item] = _to_score(e.get("score", 0))
            justifs[item] = str(e.get("justification", ""))
    if len(scores) < len(RUBRIC_ITEMS) and len(entries) == len(RUBRIC_ITEMS):
        # item names unrecognisable but the list has the right shape: use the rubric order
        scores = {item: _to_score(e.get("score", 0)) for item, e in zip(RUBRIC_ITEMS, entries)}
        justifs = {item: str(e.get("justification", "")) for item, e in zip(RUBRIC_ITEMS, entries)}
    missing = [i for i in RUBRIC_ITEMS if i not in scores]
    if missing:
        raise ValueError(f"judge response is missing rubric items: {missing}")
    return scores, justifs


# --------------------------------------------------------------------------- grading

def _request_kwargs(cfg: JudgeConfig) -> Dict[str, Any]:
    kw: Dict[str, Any] = {}
    if cfg.temperature is not None:
        kw["temperature"] = cfg.temperature
    if cfg.top_p is not None:
        kw["top_p"] = cfg.top_p
    if cfg.max_tokens is not None:
        kw["max_completion_tokens" if _is_openai(cfg) else "max_tokens"] = cfg.max_tokens
    if cfg.reasoning_effort is not None:
        kw["reasoning_effort"] = cfg.reasoning_effort
    if cfg.extra_body:
        kw["extra_body"] = dict(cfg.extra_body)
    return kw


def grade(client, cfg: JudgeConfig, problem: str, solution: str, sample_id: str = "",
          instructions: Optional[str] = None) -> Dict[str, Any]:
    """Grade one solution. Returns a grade-file row (see :data:`SCORE_COLUMNS`)."""
    row: Dict[str, Any] = {"sample_id": sample_id, "success": False, "total_score": None}
    if not str(solution or "").strip():
        row["error"] = "empty solution"
        return row
    messages = [
        {"role": "system", "content": instructions or cfg.instructions},
        {"role": "user", "content": build_user_message(problem, solution)},
    ]
    error = ""
    for attempt in range(cfg.max_attempts):
        finish = None
        try:
            resp = client.chat.completions.create(model=cfg.model, messages=messages, **_request_kwargs(cfg))
            finish = getattr(resp.choices[0], "finish_reason", None)
            content = resp.choices[0].message.content or ""
            scores, justifs = extract_scores(parse_judge_response(content))
        except Exception as e:  # noqa: BLE001 - keep the run going, record the failure
            error = f"{type(e).__name__}: {e}"
            if finish == "length":
                error = ("response truncated at the token limit (finish_reason=length): give the judge more room, "
                         "e.g. a larger --max-model-len for vLLM or a larger --max-tokens. " + error)
            if attempt + 1 < cfg.max_attempts:
                time.sleep(min(2 ** attempt, 30))
            continue
        row.update({"success": True, "total_score": sum(scores.values()), "attempts": attempt + 1})
        for item, s_col, j_col in zip(RUBRIC_ITEMS, SCORE_COLUMNS, JUSTIFICATION_COLUMNS):
            row[s_col] = scores[item]
            row[j_col] = justifs[item]
        return row
    row.update({"error": error, "attempts": cfg.max_attempts})
    return row
