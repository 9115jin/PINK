"""The 5-component, 100-point grading rubric used by PINK (paper Sec. 3.1, App. B.1, Fig. 16)."""

from __future__ import annotations

import re
from typing import Dict, List

#: Rubric components, in the order the judge is asked to report them.
RUBRIC_ITEMS: List[str] = [
    "Core Formula/Principle Identification",
    "Condition/Boundary Application",
    "Calculation Accuracy (Early-Mid)",
    "Calculation Accuracy (Late)",
    "Final Value Verification",
]

#: Maximum score of every rubric component.
POINTS_PER_ITEM: int = 20


def _slug(item: str) -> str:
    return re.sub(r"[^A-Za-z0-9-]+", "_", item).strip("_")


#: Column names used in grade files: ``score_<item>`` and ``justif_<item>``.
SCORE_COLUMNS: List[str] = [f"score_{_slug(i)}" for i in RUBRIC_ITEMS]
JUSTIFICATION_COLUMNS: List[str] = [f"justif_{_slug(i)}" for i in RUBRIC_ITEMS]

#: System prompt for the LLM judge (paper Fig. 16).
GRADING_INSTRUCTIONS: str = """# Math Exam Grading Instructions

## Role
You are a math exam grader. Evaluate the student solution using the provided rubric and grading guidelines.

## Rubric
{
  "rubric": [
    {
      "item": "Core Formula/Principle Identification",
      "points": 20,
      "criterion": "Correctly identified and stated the necessary formulas for solving the problem"
    },
    {
      "item": "Condition/Boundary Application",
      "points": 20,
      "criterion": "Accurately applied given conditions (boundaries, constraints, units, etc.)"
    },
    {
      "item": "Calculation Accuracy (Early-Mid)",
      "points": 20,
      "criterion": "Intermediate calculations were performed without omissions or contradictions"
    },
    {
      "item": "Calculation Accuracy (Late)",
      "points": 20,
      "criterion": "Final transformations and simplifications were logical and error-free"
    },
    {
      "item": "Final Value Verification",
      "points": 20,
      "criterion": "Final answer correctly incorporates all conditions and calculations"
    }
  ],
  "notes": "Points may be deducted for unit, rounding, or notation errors"
}

## Grading Guidelines
1. **Analyze each expression character by character** to identify any errors in formulas, calculations, or transformations, and provide a detailed justification explaining exactly where and why points were awarded or deducted.
2. Include specific numbers, equations, or character positions in justification when necessary (e.g., "In line 3, the expression 2x+3 should be 2x-3, where the + sign at position 3 is incorrect").
3. Total score must be an integer between 0-100.
4. Output only the JSON format below. No additional text.

## Output Format
{
  "scores": [
    {"item": "Core Formula/Principle Identification", "score": 0~20, "justification": "..."},
    {"item": "Condition/Boundary Application", "score": 0~20, "justification": "..."},
    {"item": "Calculation Accuracy (Early-Mid)", "score": 0~20, "justification": "..."},
    {"item": "Calculation Accuracy (Late)", "score": 0~20, "justification": "..."},
    {"item": "Final Value Verification", "score": 0~20, "justification": "..."}
  ],
  "total": <integer>
}"""

#: User message for the LLM judge. ``solution`` is the text being graded
#: (an OCR transcription, or the oracle transcription of the student's work).
USER_TEMPLATE: str = (
    "**Problem:**\n{problem}\n\n"
    "**Student Solution:**\n{solution}\n\n"
    "Evaluate this solution according to the rubric and output the JSON result."
)


def build_user_message(problem: str, solution: str) -> str:
    return USER_TEMPLATE.format(problem=problem, solution=solution)


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


_CANONICAL: Dict[str, str] = {_norm(i): i for i in RUBRIC_ITEMS}


def canonical_item(name: str) -> str | None:
    """Map a (possibly slightly reworded) item name returned by a judge to a rubric item."""
    key = _norm(name)
    if not key:
        return None
    if key in _CANONICAL:
        return _CANONICAL[key]
    hits = [canon for canon_key, canon in _CANONICAL.items() if canon_key in key or key in canon_key]
    return hits[0] if len(hits) == 1 else None
