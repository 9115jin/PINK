"""Offline PINK demo: no GPU, no API key.

Scores Qwen2.5-VL-7B against the oracle on 30 FERMAT samples. Both the VLM transcriptions
and the oracle transcriptions were already graded by the rubric-based LLM judge
(Qwen3-Next-80B-A3B-Thinking); the grade files are in examples/sample/.

    python examples/quickstart.py
"""

from pathlib import Path

from pink import compute_pink, load_grades, pink_table

HERE = Path(__file__).parent / "sample"

model = load_grades(HERE / "model_grades.csv")    # judge scores of the VLM transcriptions
oracle = load_grades(HERE / "oracle_grades.csv")  # judge scores of the oracle transcriptions

result = compute_pink(model, oracle)
print(result)

table = pink_table(model, oracle)
table["drop"] = table.model_total - table.penalized_total
worst = table[table.over_corrected].sort_values("drop", ascending=False).head(3)
print("\nLargest over-correction penalties (judge score of the VLM -> after penalty; oracle):")
for sid, r in worst.iterrows():
    print(f"  {sid}: {r.model_total:.0f} -> {r.penalized_total:.0f}  (oracle {r.oracle_total:.0f})")
