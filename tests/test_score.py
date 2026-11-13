import math

import numpy as np
import pandas as pd
import pytest

from pink import SCORE_COLUMNS, compute_pink, penalize, pink_table


def frame(rows):
    return pd.DataFrame(rows, columns=SCORE_COLUMNS, index=[f"s{i}" for i in range(len(rows))], dtype=float)


def test_penalty_tiers():
    oracle = [10, 10, 10, 10, 10, 10, 5]
    model = [5, 10, 11, 20, 21, 20.5, 20]
    #        keep keep  revert revert nullify nullify nullify
    assert penalize(model, oracle).tolist() == [5, 10, 10, 10, 0, 0, 0]


def test_threshold_parameter():
    assert penalize([20], [5], threshold=15).tolist() == [5]
    assert penalize([20], [5], threshold=10).tolist() == [0]


def test_paper_figure2_example():
    oracle = frame([[20, 20, 20, 5, 5]])
    model = frame([[20, 20, 20, 20, 20]])
    res = compute_pink(model, oracle)
    assert res.pink == pytest.approx(60 / 70)
    assert res.raw_ratio == pytest.approx(100 / 70)
    assert res.over_correction_rate == 1.0
    assert res.over_corrections_per_item["Calculation Accuracy (Late)"] == 1
    assert res.over_corrections_per_item["Core Formula/Principle Identification"] == 0


def test_ratio_of_means_not_mean_of_ratios():
    oracle = frame([[20, 20, 20, 20, 20], [2, 2, 2, 2, 2]])
    model = frame([[20, 20, 20, 20, 20], [1, 1, 1, 1, 1]])
    res = compute_pink(model, oracle)
    assert res.pink == pytest.approx((100 + 5) / (100 + 10))  # not (1.0 + 0.5) / 2


def test_faithful_model_scores_one():
    oracle = frame([[20, 15, 10, 5, 0], [3, 3, 3, 3, 3]])
    assert compute_pink(oracle.copy(), oracle).pink == pytest.approx(1.0)


def test_missing_samples():
    oracle = frame([[10] * 5, [10] * 5])
    model = frame([[10] * 5])
    assert compute_pink(model, oracle, missing="drop").n == 1
    zero = compute_pink(model, oracle, missing="zero")
    assert zero.n == 2 and zero.pink == pytest.approx(0.5)


def test_empty_intersection_is_nan():
    a, b = frame([[1] * 5]), frame([[1] * 5])
    b.index = ["other"]
    assert math.isnan(compute_pink(a, b).pink)


def test_pink_table_columns():
    t = pink_table(frame([[20] * 5]), frame([[10] * 5]))
    assert {"model_total", "oracle_total", "penalized_total", "over_corrected"} <= set(t.columns)
    assert t["penalized_total"].iloc[0] == 50


def test_bundled_fermat_gpt5_oracle():
    from pink import PACKAGED_ORACLES, load_grades

    assert "fermat:gpt-5" in PACKAGED_ORACLES
    oracle = load_grades("fermat:gpt-5")
    assert len(oracle) == 2244 and list(oracle.columns) == SCORE_COLUMNS
    assert oracle.sum(axis=1).mean() == pytest.approx(71.3, abs=0.05)  # paper Sec. 4: perturbed solutions average 71.3
    assert compute_pink(oracle.copy(), "fermat:gpt-5").pink == pytest.approx(1.0)
