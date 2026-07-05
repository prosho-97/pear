"""Command line interface for PEAR."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from pear.inference import load_metric, score_pairwise, score_reference_anchored
from pear.mbr import pear_utility_matrix, select_mbr_hypothesis


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="pear", description="Run PEAR MT metric inference."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    score = sub.add_parser("score", help="Score TSV rows with src/mt_0/mt_1 columns.")
    _add_model_args(score)
    score.add_argument("--input", type=Path, required=True)
    score.add_argument("--output", type=Path)
    score.add_argument("--mode", choices=["pairwise", "reference"], default="pairwise")
    score.add_argument("--both", action="store_true", help="Score both pair orders.")
    score.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of pairwise examples per PEAR forward batch.",
    )
    score.add_argument(
        "--gpus",
        type=_parse_gpus,
        default="auto",
        help=(
            "GPU usage: auto uses one available CUDA/MPS device, 0 forces CPU, "
            "and 1 forces one accelerator device."
        ),
    )

    mbr = sub.add_parser("mbr", help="Run PEAR MBR over JSONL n-best lists.")
    _add_model_args(mbr)
    mbr.add_argument(
        "--input",
        type=Path,
        required=True,
        help="JSONL with src and hypotheses fields.",
    )
    mbr.add_argument("--output", type=Path)
    mbr.add_argument("--utility", choices=["full", "half"], default="half")
    mbr.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of pairwise utility comparisons per PEAR forward batch.",
    )
    mbr.add_argument(
        "--gpus",
        type=_parse_gpus,
        default="auto",
        help=(
            "GPU usage: auto uses one available CUDA/MPS device, 0 forces CPU, "
            "and 1 forces one accelerator device."
        ),
    )

    args = parser.parse_args()
    if args.command == "score":
        _score(args)
    elif args.command == "mbr":
        _mbr(args)


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        default="pear",
        help=(
            "Model to load: pear, pear-xl, a Hugging Face repo id, or a local "
            ".ckpt path. Defaults to pear."
        ),
    )
    parser.add_argument(
        "--hf-model",
        help=(
            "Explicit Hugging Face repo id or alias to load. This is a clearer "
            "alternative to passing the same value through --model."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help=(
            "Explicit local Lightning .ckpt path to load. This is a clearer "
            "alternative to passing the same path through --model."
        ),
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true")


def _parse_gpus(value: str) -> int | str:
    if value == "auto":
        return value
    try:
        gpus = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected auto, 0, or 1") from exc
    if gpus not in {0, 1}:
        raise argparse.ArgumentTypeError("expected auto, 0, or 1")
    return gpus


def _resolve_model_arg(args: argparse.Namespace) -> str | Path:
    selected = [args.checkpoint is not None, args.hf_model is not None]
    if sum(selected) > 1:
        raise SystemExit("Use only one of --checkpoint or --hf-model.")
    if args.checkpoint is not None:
        return args.checkpoint
    if args.hf_model is not None:
        return args.hf_model
    return args.model


def _score(args: argparse.Namespace) -> None:
    model = load_metric(
        _resolve_model_arg(args),
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    rows = list(csv.DictReader(args.input.open(newline=""), delimiter="\t"))
    sources = [row["src"] for row in rows]
    mode = "both" if args.both else "single"
    if args.mode == "pairwise":
        scores = score_pairwise(
            model,
            sources,
            [r["mt_0"] for r in rows],
            [r["mt_1"] for r in rows],
            mode=mode,
            batch_size=args.batch_size,
            gpus=args.gpus,
        )
    else:
        scores = score_reference_anchored(
            model,
            sources,
            [r["mt"] for r in rows],
            [r["ref"] for r in rows],
            mode=mode,
            batch_size=args.batch_size,
            gpus=args.gpus,
        )
    if isinstance(scores, dict):
        for key, values in scores.items():
            for row, value in zip(rows, values):
                row[f"pear_{key}"] = value
    else:
        for row, value in zip(rows, scores):
            row["pear_score"] = value
    fieldnames = list(rows[0]) if rows else []
    output = args.output or args.input.with_suffix(".pear.tsv")
    with output.open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _mbr(args: argparse.Namespace) -> None:
    model = load_metric(
        _resolve_model_arg(args),
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    output = args.output or args.input.with_suffix(".pear-mbr.jsonl")
    with args.input.open() as src, output.open("w") as out:
        for line in src:
            item = json.loads(line)
            matrix = pear_utility_matrix(
                model,
                item["src"],
                item["hypotheses"],
                mode=args.utility,
                batch_size=args.batch_size,
                gpus=args.gpus,
            )
            index, utility = select_mbr_hypothesis(matrix)
            item.update(
                {
                    "selected_index": index,
                    "selected": item["hypotheses"][index],
                    "utility": utility,
                }
            )
            out.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
