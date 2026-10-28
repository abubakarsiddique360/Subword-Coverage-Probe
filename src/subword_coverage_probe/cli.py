"""Command line entry point for subword-probe."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .bpe import BpeError, decode, encode, train
from .coverage import (
    most_common_tokens,
    oov_rate,
    score_bpe,
    score_chars,
    score_words,
    unused_merges,
)
from .report import (
    render_json,
    render_merges,
    render_score,
    render_sweep,
    render_table,
    render_tokens,
)

DEFAULT_CORPUS = "examples/corpus.txt"


def _read(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise BpeError(f"cannot read {path}: {exc}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="subword-probe",
        description="Train a BPE tokenizer from scratch and measure what it costs per word.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    tr = sub.add_parser("train", help="learn merges and show the first ones")
    tr.add_argument("--corpus", default=DEFAULT_CORPUS)
    tr.add_argument("--vocab", type=int, default=400)
    tr.add_argument("--merges-to-show", type=int, default=20)

    enc = sub.add_parser("encode", help="tokenize a string")
    enc.add_argument("--corpus", default=DEFAULT_CORPUS)
    enc.add_argument("--vocab", type=int, default=400)
    enc.add_argument("--text", required=True)

    cmp_cmd = sub.add_parser("compare", help="char, word and BPE tokenizers on the same text")
    cmp_cmd.add_argument("--corpus", default=DEFAULT_CORPUS)
    cmp_cmd.add_argument("--heldout", default="examples/heldout.txt")
    cmp_cmd.add_argument("--vocab", type=int, default=400)
    cmp_cmd.add_argument("--format", choices=("text", "json"), default="text")

    sweep = sub.add_parser("sweep", help="cost against vocabulary size")
    sweep.add_argument("--corpus", default=DEFAULT_CORPUS)
    sweep.add_argument("--heldout", default="examples/heldout.txt")
    sweep.add_argument("--vocabs", default="300,400,600,800,1200,1600,2400")
    sweep.add_argument("--from-here", action="store_true",
                       help="only encode below 26000 vocab (speed guard)")

    common = sub.add_parser("common", help="most frequent tokens in the text")
    common.add_argument("--corpus", default=DEFAULT_CORPUS)
    common.add_argument("--vocab", type=int, default=400)
    common.add_argument("--limit", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        corpus = _read(args.corpus)
    except BpeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.command == "train":
        try:
            model = train(corpus, args.vocab)
        except BpeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(render_merges(model, args.merges_to_show))
        print()
        print(render_score(score_bpe(f"bpe-{model.size}", corpus, model)))
        print()
        print(f"unused merges on this corpus: {unused_merges(model, corpus)}")
        return 0

    if args.command == "encode":
        try:
            model = train(corpus, args.vocab)
        except BpeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        tokens = encode(args.text, model)
        print(render_tokens(tokens))
        print()
        print(f"{len(tokens)} tokens for {len(args.text.split())} words")
        print(f"round trip: {decode(tokens)!r}")
        return 0

    if args.command == "compare":
        try:
            heldout = _read(args.heldout)
            model = train(corpus, args.vocab)
        except BpeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        scores = [
            score_chars(heldout),
            score_words(heldout),
            score_bpe(f"bpe-{model.size}", heldout, model),
        ]
        if args.format == "json":
            print(render_json([score.as_dict() for score in scores]))
        else:
            print(render_table(scores))
            print()
            print(render_score(scores[2]))
            print()
            print(f"oov rate (bpe): {oov_rate(model, heldout.split()) * 100:.0f}%")
        return 0

    if args.command == "sweep":
        try:
            heldout = _read(args.heldout)
        except BpeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        rows = []
        seen: set[int] = set()
        for raw in args.vocabs.split(","):
            if not raw.strip():
                continue
            size = int(raw)
            model = train(corpus, size)
            if model.size in seen:
                continue
            seen.add(model.size)
            score = score_bpe(f"bpe-{model.size}", heldout, model)
            rows.append(
                {
                    **score.as_dict(),
                    "vocab": model.size,
                    "merges_fired": len(model.merges) - unused_merges(model, heldout),
                    "requested": size,
                }
            )
        print(render_sweep(rows))
        requested = [int(part) for part in args.vocabs.split(",") if part.strip()]
        if rows and requested and rows[-1]["vocab"] < max(requested):
            print()
            print(
                f"note: the corpus has no adjacent pairs left above {rows[-1]['vocab']} symbols, "
                "so larger vocabularies are impossible without more text"
            )
        return 0

    if args.command == "common":
        try:
            model = train(corpus, args.vocab)
        except BpeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        for token, count in most_common_tokens(corpus, model, args.limit):
            print(f"{count:>5}  {token}")
        return 0

    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
