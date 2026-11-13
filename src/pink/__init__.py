"""PINK: Penalized INK-based score for faithful handwritten-math OCR evaluation.

Seong et al., "When VLMs 'Fix' Students: Identifying and Penalizing Over-Correction in
the Evaluation of Multi-line Handwritten Math OCR", EMNLP 2026.
"""

__version__ = "1.0.0"

from .judge import JUDGE_PRESETS, JudgeConfig, grade, judge_from_preset
from .rubric import GRADING_INSTRUCTIONS, RUBRIC_ITEMS, SCORE_COLUMNS
from .score import DEFAULT_THRESHOLD, PINKResult, compute_pink, load_grades, penalize, pink_table

__all__ = [
    "__version__",
    "DEFAULT_THRESHOLD",
    "GRADING_INSTRUCTIONS",
    "JUDGE_PRESETS",
    "JudgeConfig",
    "PINKResult",
    "RUBRIC_ITEMS",
    "SCORE_COLUMNS",
    "compute_pink",
    "grade",
    "judge_from_preset",
    "load_grades",
    "penalize",
    "pink_table",
]
