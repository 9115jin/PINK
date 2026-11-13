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
