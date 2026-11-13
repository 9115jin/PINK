"""VLM transcription of handwritten math images (paper Sec. 4, prompt in Fig. 18).

Works with any OpenAI-compatible vision endpoint (OpenAI, Gemini's OpenAI-compatible
API, or an open model served with ``vllm serve``). You can also transcribe with your own
stack and skip this module entirely; ``pink grade`` only needs the resulting text.
"""

from __future__ import annotations

import base64
import mimetypes
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

SYSTEM_PROMPT = (
    "You are a math assistant specializing in extracting mathematical question and answer content from "
    "handwritten images of math problems by middle or high school students. Your task is to analyze the given "
    "Image, which contains the handwritten math Question-Answer pair, and convert it into LaTeX format. Follow "
    "the specific instructions provided carefully and output your response strictly in the requested format. "
    "Ensure that your evaluation is precise, accurate, and based solely on the information visible in the image."
)

USER_PROMPT = """The image provided contains a handwritten math problem with both a Question and an Answer at a middle or high school level. Your task is to explicitly perform OCR on the handwritten text and extract the content in LaTeX format. Return only the extracted content, formatted in LaTeX, exactly as it appears in the image, in the **Question:** and **Answer:** fields.

Ensure that no extra information is added that is not in the image.

Please return the LaTeX output as follows:
**Question:**<Extracted Question text in LaTeX>
**Answer:**<Extracted Answer text in LaTeX>"""

# Prompt with explicit faithfulness instructions (paper Fig. 19, App. F). In the paper it
# reduced over-correction only slightly and did not improve PINK on average.
FAITHFUL_SYSTEM_PROMPT = (
    "You are a math assistant specializing in extracting mathematical question and answer content from "
    "handwritten images of math problems by middle or high school students. Your task is to analyze the given "
    "Image, which contains the handwritten math Question-Answer pair, and convert it into LaTeX format. Follow "
    "the specific instructions provided carefully and output your response strictly in the requested format. "
    "Ensure that your transcription is precise, accurate, and based solely on the information visible in the "
    "image. IMPORTANT: You must transcribe EXACTLY what is written, including any errors or mistakes made by "
    "the student. Do NOT correct or fix any mathematical errors."
)

FAITHFUL_USER_PROMPT = """The image provided contains a handwritten math problem with both a Question and an Answer at a middle or high school level. Your task is to explicitly perform OCR on the handwritten text and extract the content in LaTeX format.

CRITICAL INSTRUCTIONS FOR FAITHFUL TRANSCRIPTION:
- Transcribe EXACTLY what is written in the image, including any mathematical errors or mistakes.
- Do NOT correct, fix, or improve any errors in the student's work.
- If the student wrote "2 + 2 = 5", you must transcribe it as "2 + 2 = 5", NOT "2 + 2 = 4".
- Your goal is FAITHFUL TRANSCRIPTION, not mathematical correction.
- Preserve all calculation errors, wrong answers, and mistakes exactly as they appear.

Return only the extracted content, formatted in LaTeX, exactly as it appears in the image, in the **Question:** and **Answer:** fields.

Ensure that no extra information is added that is not in the image.

Please return the LaTeX output as follows:
**Question:**<Extracted Question text in LaTeX>
**Answer:**<Extracted Answer text in LaTeX>"""

PROMPTS: Dict[str, Tuple[str, str]] = {
    "default": (SYSTEM_PROMPT, USER_PROMPT),
    "faithful": (FAITHFUL_SYSTEM_PROMPT, FAITHFUL_USER_PROMPT),
}


def clean_transcription(text: Any) -> str:
    """Strip markdown fences and the **Question:** / **Answer:** labels from a raw VLM output."""
    text = "" if text is None else str(text)
    text = re.sub(r"^\s*```[a-zA-Z]*\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    text = re.sub(r"\*\*(Question|Answer):\*\*[ \t]*", "", text)
    text = re.sub(r"\\textbf\{(Question|Answer)\s*:?\s*\}:?[ \t]*", "", text)
    text = re.sub(r"^[ \t]*(Question|Answer):[ \t]*", "", text, flags=re.MULTILINE)
    return text.strip()


def image_to_data_url(image_path) -> str:
    mime = mimetypes.guess_type(str(image_path))[0] or "image/png"
    with open(image_path, "rb") as f:
        return f"data:{mime};base64,{base64.b64encode(f.read()).decode()}"


def make_client(base_url: Optional[str] = None, api_key: Optional[str] = None, timeout: float = 600.0):
    from openai import OpenAI

    key = api_key or os.environ.get("OPENAI_API_KEY") or ("EMPTY" if base_url else None)
    return OpenAI(api_key=key, base_url=base_url, timeout=timeout)


def transcribe(client, model: str, image_path, prompt: str = "default", max_tokens: Optional[int] = 4096,
               temperature: Optional[float] = 0.0) -> str:
    """Transcribe one image and return the raw model output."""
    system, user = PROMPTS[prompt]
    kwargs: Dict[str, Any] = {}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if temperature is not None:
        kwargs["temperature"] = temperature
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "text", "text": user},
                {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
            ]},
        ],
        **kwargs,
    )
    return resp.choices[0].message.content or ""


def transcribe_row(client, model: str, item: Dict[str, Any], image_dir, attempts: int = 3,
                   **kwargs: Any) -> Dict[str, Any]:
    """Transcribe the image of one manifest item; returns a row ready for ``pink grade``."""
    row = {"sample_id": item["sample_id"], "problem": item.get("problem", ""), "solution": "",
           "raw_output": "", "success": False}
    image_path = Path(image_dir) / item["image"]
    error = ""
    for _ in range(attempts):
        try:
            raw = transcribe(client, model, image_path, **kwargs)
        except Exception as e:  # noqa: BLE001
            error = f"{type(e).__name__}: {e}"
            continue
        row.update(raw_output=raw, solution=clean_transcription(raw), success=bool(raw.strip()))
        if row["success"]:
            return row
        error = "empty output"
    row["error"] = error
    return row
