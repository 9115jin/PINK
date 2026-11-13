"""Command-line interface: ``pink fermat | ocr | grade | score``."""

from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path
from typing import List, Optional

from . import __version__


def _limit(items: list, limit: Optional[int]) -> list:
    return items if limit is None else items[:limit]


def _check_endpoint(client, base_url: Optional[str], model: str) -> None:
    """Fail fast if a self-hosted endpoint is down or does not serve ``model``."""
    if not base_url or "api.openai.com" in base_url:
        return
    try:
        served = [m.id for m in client.models.list().data]
    except Exception as e:  # noqa: BLE001
        sys.exit(f"cannot reach {base_url} ({type(e).__name__}). Is the server up? "
                 "Check with: curl {}/models".format(base_url.rstrip("/")))
    if model not in served:
        sys.exit(f"{base_url} does not serve {model!r}; it serves {served}. "
                 "Pass --model, or start the server with --served-model-name.")


def cmd_fermat(args: argparse.Namespace) -> int:
    from .data import export_fermat

    manifest = export_fermat(args.out_dir, limit=args.limit, cache_dir=args.cache_dir, seed=args.seed,
                             images=not args.text_only)
    n = sum(1 for _ in open(manifest, encoding="utf-8"))
    print(f"wrote {n} samples -> {manifest}")
    print(f"next: transcribe the images with your VLM, e.g.\n  pink ocr --manifest {manifest} --model <vlm> "
          "--base-url <url> --out ocr.jsonl")
    return 0


def cmd_ocr(args: argparse.Namespace) -> int:
    from ._runner import read_jsonl, run_jobs
    from .ocr import make_client, transcribe_row

    items = _limit(read_jsonl(args.manifest), args.limit)
    image_dir = args.image_dir or Path(args.manifest).parent
    client = make_client(args.base_url, args.api_key)
    _check_endpoint(client, args.base_url, args.model)
    fn = partial(transcribe_row, client, args.model, image_dir=image_dir, prompt=args.prompt,
                 max_tokens=args.max_tokens, temperature=args.temperature)
    stats = run_jobs(items, fn, args.out, workers=args.workers, resume=not args.overwrite, desc=f"ocr:{args.model}")
    print(json.dumps(stats))
    return 0 if stats["failed"] == 0 else 1


def cmd_grade(args: argparse.Namespace) -> int:
    from ._runner import read_jsonl, run_jobs
    from .judge import JudgeConfig, grade, judge_from_preset, make_client
    from .rubric import JUSTIFICATION_COLUMNS, SCORE_COLUMNS

    if args.judge:
        cfg = judge_from_preset(args.judge, model=args.model, base_url=args.base_url, api_key=args.api_key,
                                temperature=args.temperature, max_tokens=args.max_tokens,
                                reasoning_effort=args.reasoning_effort)
    elif args.model:
        cfg = JudgeConfig(model=args.model, base_url=args.base_url, api_key=args.api_key,
                          temperature=args.temperature, max_tokens=args.max_tokens,
                          reasoning_effort=args.reasoning_effort)
    else:
        sys.exit("pass --judge {gpt-5,qwen3-next-80b} or --model (with --base-url for a local server)")

    items = [it for it in read_jsonl(args.input) if "solution" in it]
    items = _limit(items, args.limit)
    client = make_client(cfg)
    _check_endpoint(client, cfg.base_url, cfg.model)

    def fn(item):
        return grade(client, cfg, item.get("problem", ""), item.get("solution", ""), str(item["sample_id"]))

    columns = ["sample_id", "success", "total_score", *SCORE_COLUMNS, *JUSTIFICATION_COLUMNS, "attempts", "error"]
    out = Path(args.out)
    stats = run_jobs(items, fn, out, columns=columns if out.suffix == ".csv" else None,
                     workers=args.workers, resume=not args.overwrite, desc=f"grade:{cfg.model}")
    print(json.dumps(stats))
    return 0 if stats["failed"] == 0 else 1


def cmd_score(args: argparse.Namespace) -> int:
    from .score import compute_pink, load_grades, pink_table

    model, oracle = load_grades(args.model), load_grades(args.oracle)
    result = compute_pink(model, oracle, threshold=args.threshold, missing=args.missing)
    if args.per_sample:
        pink_table(model, oracle, threshold=args.threshold, missing=args.missing).to_csv(args.per_sample)
    print(json.dumps(result.to_dict(), indent=2) if args.json else result)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pink", description="PINK: penalized, over-correction-aware evaluation "
                                "of handwritten math OCR (EMNLP 2026).")
    p.add_argument("--version", action="version", version=f"pink {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    f = sub.add_parser("fermat", help="download FERMAT and write images/ + manifest.jsonl (oracle transcriptions)")
    f.add_argument("--out-dir", required=True)
    f.add_argument("--limit", type=int, help="export a random subset of N samples (for a quick try)")
    f.add_argument("--seed", type=int, default=0, help="seed for the --limit subset")
    f.add_argument("--text-only", action="store_true", help="skip the images (enough to grade the oracle)")
    f.add_argument("--cache-dir", help="Hugging Face datasets cache directory")
    f.set_defaults(func=cmd_fermat)

    o = sub.add_parser("ocr", help="transcribe images with a VLM behind an OpenAI-compatible API")
    o.add_argument("--manifest", required=True, help="manifest.jsonl from `pink fermat` (or your own)")
    o.add_argument("--image-dir", help="directory the manifest's image paths are relative to (default: its folder)")
    o.add_argument("--model", required=True, help="model name as served by the endpoint")
    o.add_argument("--base-url", help="e.g. http://localhost:8000/v1 for `vllm serve` (default: OpenAI API)")
    o.add_argument("--api-key")
    o.add_argument("--prompt", choices=["default", "faithful"], default="default")
    o.add_argument("--max-tokens", type=int, default=4096)
    o.add_argument("--temperature", type=float, default=0.0)
    o.add_argument("--workers", type=int, default=8)
    o.add_argument("--limit", type=int)
    o.add_argument("--overwrite", action="store_true", help="start from scratch instead of resuming")
    o.add_argument("--out", required=True, help="output .jsonl (input for `pink grade`)")
    o.set_defaults(func=cmd_ocr)

    g = sub.add_parser("grade", help="grade solutions with the rubric-based LLM judge")
    g.add_argument("--input", required=True, help="jsonl with sample_id, problem, solution")
    g.add_argument("--out", required=True, help="output grades (.csv or .jsonl)")
    g.add_argument("--judge", choices=["gpt-5", "qwen3-next-80b"], help="paper judge preset")
    g.add_argument("--model", help="judge model name (overrides the preset)")
    g.add_argument("--base-url", help="OpenAI-compatible endpoint of the judge")
    g.add_argument("--api-key")
    g.add_argument("--temperature", type=float)
    g.add_argument("--max-tokens", type=int)
    g.add_argument("--reasoning-effort", choices=["minimal", "low", "medium", "high"])
    g.add_argument("--workers", type=int, default=8)
    g.add_argument("--limit", type=int)
    g.add_argument("--overwrite", action="store_true", help="start from scratch instead of resuming")
    g.set_defaults(func=cmd_grade)

    s = sub.add_parser("score", help="compute PINK from model and oracle grades")
    s.add_argument("--model", required=True, help="grades of the VLM transcriptions")
    s.add_argument("--oracle", required=True,
                   help="grades of the oracle transcriptions from the same judge, or a bundled set: "
                        "fermat:gpt-5 or fermat:qwen3-next-80b")
    s.add_argument("--threshold", type=float, default=10.0, help="minor/major boundary T (default 10)")
    s.add_argument("--missing", choices=["drop", "zero"], default="drop",
                   help="oracle samples without a model grade: drop them (paper) or score them 0")
    s.add_argument("--per-sample", metavar="CSV", help="also write per-sample penalized scores")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_score)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
