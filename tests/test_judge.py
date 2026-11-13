import json
from types import SimpleNamespace

import pytest

from pink.judge import JudgeConfig, extract_scores, grade, judge_from_preset, parse_judge_response
from pink.rubric import RUBRIC_ITEMS, SCORE_COLUMNS, canonical_item

GOOD = {"scores": [{"item": i, "score": s, "justification": "ok"} for i, s in zip(RUBRIC_ITEMS, [20, 15, 10, 5, 0])],
        "total": 50}


def test_parse_plain_fenced_and_thinking():
    assert parse_judge_response(json.dumps(GOOD))["total"] == 50
    assert parse_judge_response("```json\n" + json.dumps(GOOD) + "\n```")["total"] == 50
    noisy = "Let me think {draft}... </think>\nHere you go: " + json.dumps(GOOD) + " done"
    assert parse_judge_response(noisy)["total"] == 50
    assert parse_judge_response("<think>{\"scores\": []}</think>" + json.dumps(GOOD))["total"] == 50


def test_parse_trailing_commas():
    text = json.dumps(GOOD)[:-1] + ",}"
    assert parse_judge_response("Result: " + text)["total"] == 50


def test_parse_failure():
    with pytest.raises(ValueError):
        parse_judge_response("no json here")


def test_extract_scores_clamps_and_matches_names():
    data = {"scores": [
        {"item": "core formula / principle identification", "score": "18/20", "justification": ""},
        {"item": "Condition/Boundary Application", "score": 25, "justification": ""},
        {"item": "Calculation Accuracy (Early-Mid)", "score": -3, "justification": ""},
        {"item": "Calculation Accuracy (Late)", "score": 12.6, "justification": ""},
        {"item": "Final Value Verification", "score": 7, "justification": ""},
    ]}
    scores, _ = extract_scores(data)
    assert [scores[i] for i in RUBRIC_ITEMS] == [18, 20, 0, 13, 7]


def test_extract_scores_positional_fallback_and_missing():
    data = {"scores": [{"item": f"x{k}", "score": k} for k in range(5)]}
    scores, _ = extract_scores(data)
    assert [scores[i] for i in RUBRIC_ITEMS] == [0, 1, 2, 3, 4]
    with pytest.raises(ValueError):
        extract_scores({"scores": GOOD["scores"][:4]})


def test_ambiguous_item_name_is_not_guessed():
    assert canonical_item("Calculation Accuracy") is None
    assert canonical_item("Final Value") == "Final Value Verification"


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])


def test_grade_retries_then_succeeds():
    client = FakeClient(["garbage", json.dumps(GOOD)])
    cfg = JudgeConfig(model="m", base_url="http://x/v1", max_attempts=3)
    row = grade(client, cfg, "P", "S", "id1")
    assert row["success"] and row["attempts"] == 2 and row["total_score"] == 50
    assert [row[c] for c in SCORE_COLUMNS] == [20, 15, 10, 5, 0]
    msgs = client.calls[0]["messages"]
    assert msgs[0]["role"] == "system" and "**Student Solution:**\nS" in msgs[1]["content"]


def test_grade_records_failure_and_empty_solution():
    cfg = JudgeConfig(model="m", base_url="http://x/v1", max_attempts=1)
    row = grade(FakeClient([RuntimeError("boom")]), cfg, "P", "S", "id1")
    assert not row["success"] and "boom" in row["error"]
    assert grade(FakeClient([]), cfg, "P", "   ", "id2")["error"] == "empty solution"


def test_request_parameters_per_backend():
    client = FakeClient([json.dumps(GOOD)])
    grade(client, judge_from_preset("gpt-5", max_tokens=100), "P", "S")
    kw = client.calls[0]
    assert kw["model"] == "gpt-5-2025-08-07" and kw["reasoning_effort"] == "medium"
    assert kw["max_completion_tokens"] == 100 and "temperature" not in kw

    client = FakeClient([json.dumps(GOOD)])
    grade(client, judge_from_preset("qwen3-next-80b"), "P", "S")
    kw = client.calls[0]
    assert kw["temperature"] == 0.6 and kw["top_p"] == 0.95 and "max_tokens" not in kw


def test_truncated_response_gets_actionable_error():
    client = FakeClient([])
    truncated = SimpleNamespace(choices=[SimpleNamespace(finish_reason="length",
                                                         message=SimpleNamespace(content="Okay, let's think..."))])
    client.chat.completions.create = lambda **kw: truncated
    row = grade(client, JudgeConfig(model="m", base_url="http://x/v1", max_attempts=1), "P", "S")
    assert not row["success"] and "finish_reason=length" in row["error"] and "max-model-len" in row["error"]
