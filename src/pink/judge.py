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
    #: HTTP timeout per request in seconds. Local thinking judges can need well over 10 min per request under load;
    #: a too-short timeout silently biases the kept grades towards short reasoning. Default: $PINK_JUDGE_TIMEOUT or 600.
    timeout: float = field(default_factory=lambda: float(os.environ.get("PINK_JUDGE_TIMEOUT", "600")))


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
    # Light open-source judge (2026-11): fits on one 24-48 GB GPU. On all 2,244 FERMAT samples x 15 paper VLMs it ranks
    # the models like GPT-5 (Kendall tau = 0.98, r(PINK) = 0.997, sample-level r = 0.90). Thinking must stay on:
    # without it the judge scores almost only 0 or 20 per component and agrees less with GPT-5.
    # Sampling follows the Qwen3.5 generation config (T=1.0, top-p 0.95, top-k 20).
    "qwen3.5-9b": JudgeConfig(
        model="Qwen/Qwen3.5-9B",
        base_url="http://localhost:8000/v1",
        temperature=1.0,
        top_p=0.95,
        extra_body={"chat_template_kwargs": {"enable_thinking": True}, "top_k": 20},
        timeout=7200.0,
    ),
    # Closest open judge to GPT-5 (2026-11): needs two 48 GB GPUs (TP=2). On all 2,244 FERMAT samples x 15 paper VLMs:
    # Kendall tau = 0.98, r(PINK) = 0.998, sample-level r = 0.93, and over-correction rates within a few points of
    # GPT-5's (the 9B judge's run 3-8 points low). Same sampling and thinking setup as qwen3.5-9b.
    "qwen3.8-27b": JudgeConfig(
        model="Qwen/Qwen3.8-27B",
        base_url="http://localhost:8000/v1",
        temperature=1.0,
        top_p=0.95,
        extra_body={"chat_template_kwargs": {"enable_thinking": True}, "top_k": 20},
        timeout=7200.0,
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

_JSON_ESCAPE = re.compile(r'\\(["\\/bfnrt]|u[0-9a-fA-F]{4})?')


def _loads_lenient(s: str) -> Any:
    """``json.loads`` that also accepts backslashes a judge forgot to escape.

    Small judges often copy LaTeX into the justifications verbatim (``\\(``, ``\\{``, ``\\implies``), which is
    invalid JSON. Only when strict parsing fails, every backslash that does not start a valid JSON escape is escaped.
    """
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    fixed = _JSON_ESCAPE.sub(lambda m: m.group(0) if m.group(1) else "\\\\", s)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        return json.loads(fixed, strict=False)  # raw newlines / tabs inside strings


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
            return _loads_lenient(block)
        except json.JSONDecodeError:
            continue
    try:
        return _loads_lenient(text)
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
                return _loads_lenient(attempt)
            except json.JSONDecodeError:
                continue
    fallback = _scores_by_regex(text)
    if fallback is not None:
        return fallback
    raise ValueError(f"could not parse JSON from judge response: {text[:200]!r}")


_ITEM_SCORE = re.compile(r'"item"\s*:\s*"([^"]+)"\s*,\s*"score"\s*:\s*"?(-?\d+(?:\.\d+)?)')


def _scores_by_regex(text: str) -> Optional[Dict[str, Any]]:
    """Last resort for structurally broken JSON (e.g. a missing ``}`` after the last item, a frequent slip of small
    judges): read the ``"item": ..., "score": ...`` pairs directly. Only used when all five rubric items are found."""
    found: Dict[str, Any] = {}
    for name, score in _ITEM_SCORE.findall(text):
        item = canonical_item(name)
        if item is not None and item not in found:
            found[item] = float(score)
    if len(found) < len(RUBRIC_ITEMS):
        return None
    return {"scores": [{"item": i, "score": found[i], "justification": ""} for i in RUBRIC_ITEMS],
            "parsed_by": "regex fallback"}


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


def _encodable(text: str) -> str:
    """Drop lone UTF-16 surrogates (from ``\\ud835``-style escapes in the judge's JSON); they cannot be written as UTF-8."""
    return text.encode("utf-8", "replace").decode("utf-8")


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
            row[j_col] = _encodable(justifs[item])
        return row
    row.update({"error": error, "attempts": cfg.max_attempts})
    return row
