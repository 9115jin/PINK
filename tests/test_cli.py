import json
from pathlib import Path

from pink.cli import main

SAMPLE = Path(__file__).resolve().parents[1] / "examples" / "sample"


def test_score_on_bundled_sample(capsys, tmp_path):
    per_sample = tmp_path / "per_sample.csv"
    assert main(["score", "--model", str(SAMPLE / "model_grades.csv"), "--oracle", str(SAMPLE / "oracle_grades.csv"),
                 "--json", "--per-sample", str(per_sample)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert round(out["pink"], 4) == 0.74 and out["n"] == 30
    assert per_sample.exists()


def test_grade_cli_light_judge_options(tmp_path, monkeypatch):
    import pink._runner as runner
    import pink.judge as judge
    from pink import cli

    seen = {}
    monkeypatch.setattr(judge, "make_client", lambda cfg: seen.setdefault("cfg", cfg))
    monkeypatch.setattr(cli, "_check_endpoint", lambda *a, **k: None)
    monkeypatch.setattr(runner, "run_jobs", lambda *a, **k: {"failed": 0})
    inp = tmp_path / "in.jsonl"
    inp.write_text('{"sample_id": "s1", "problem": "p", "solution": "s"}\n')
    rc = cli.main(["grade", "--judge", "qwen3.5-9b", "--input", str(inp), "--out", str(tmp_path / "g.csv"),
                   "--extra-body", '{"chat_template_kwargs": {"enable_thinking": false}}', "--timeout", "99",
                   "--top-p", "0.8"])
    cfg = seen["cfg"]
    assert rc == 0
    assert cfg.extra_body == {"chat_template_kwargs": {"enable_thinking": False}, "top_k": 20}
    assert (cfg.timeout, cfg.top_p, cfg.temperature) == (99.0, 0.8, 1.0)
    # the preset itself is untouched
    assert judge.JUDGE_PRESETS["qwen3.5-9b"].extra_body["chat_template_kwargs"]["enable_thinking"] is True
