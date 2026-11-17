from pink.data import oracle_transcription
from pink.ocr import clean_transcription


def test_oracle_removes_red_markers_and_crlf():
    raw = "a = 2\r\n\\textcolor{red}{b = 5} and \\textcolor{red}{c}\r\n"
    assert oracle_transcription(raw) == "a = 2\nb = 5 and c\n"


def test_oracle_keeps_nested_latex_and_other_colours():
    assert oracle_transcription(r"x = \textcolor{red}{\frac{1}{2}} + 1") == r"x = \frac{1}{2} + 1"
    assert oracle_transcription(r"\textcolor{green}{y}") == r"\textcolor{green}{y}"
    assert oracle_transcription(r"\textcolor{red}{unclosed") == "unclosed"


def test_clean_transcription():
    raw = "```latex\n**Question:** Solve $x+1=2$.\n**Answer:** $x=1$\n```"
    assert clean_transcription(raw) == "Solve $x+1=2$.\n$x=1$"
    assert clean_transcription(None) == ""


def test_clean_transcription_drops_reasoning():
    assert clean_transcription("<think>2+2 is 4, but she wrote 5</think>\n**Answer:** 2+2=5") == "2+2=5"
    assert clean_transcription("hmm let me read...</think>**Answer:** x=1") == "x=1"
    assert clean_transcription("<think>never closed") == ""


def test_clean_transcription_drops_yaml_front_matter():
    raw = "---\nprimary_language: en\nis_rotation_valid: True\n---\nSolve $x+1=2$."
    assert clean_transcription(raw) == "Solve $x+1=2$."
    assert clean_transcription("a\n---\nb: c\n---\nd").startswith("a")  # only a leading block is removed


def test_custom_prompts_and_image_order():
    from types import SimpleNamespace
    from pink.ocr import transcribe
    calls = []
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kw: calls.append(kw) or SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="x"))]))))
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(b"\x89PNG\r\n\x1a\n")
    transcribe(client, "m", f.name, system_prompt="", user_prompt="OCR:", image_first=True)
    os.unlink(f.name)
    msgs = calls[0]["messages"]
    assert [m["role"] for m in msgs] == ["user"]
    assert [p["type"] for p in msgs[0]["content"]] == ["image_url", "text"] and msgs[0]["content"][1]["text"] == "OCR:"
