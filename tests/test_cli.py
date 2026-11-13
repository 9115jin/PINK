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
