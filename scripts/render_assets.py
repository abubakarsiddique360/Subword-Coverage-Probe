"""Regenerate README assets from real runs.

Usage:  python scripts/render_assets.py
Writes: docs/results.json, docs/images/compare.png,
        docs/images/fertility-curve.svg, docs/images/segmentation.png
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from subword_coverage_probe.bpe import END_OF_WORD, encode, train  # noqa: E402
from subword_coverage_probe.coverage import (  # noqa: E402
    score_bpe,
    score_chars,
    score_words,
    unused_merges,
)
from subword_coverage_probe.report import render_score, render_sweep, render_table  # noqa: E402

BG = "#0d1117"
CHROME = "#161b22"
FG = "#e6edf3"
MUTED = "#8b949e"
ACCENT = "#79c0ff"
VOCABS = (300, 400, 600, 800, 883)
SENTENCE = "the decoder hunts for the flag byte and proves each candidate frame with a checksum"


def load_font(size: int) -> ImageFont.FreeTypeFont:
    for name in ("consola.ttf", "cour.ttf", "DejaVuSansMono.ttf"):
        candidate = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / name
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def render_terminal(text: str, out_path: Path, title: str) -> None:
    font = load_font(15)
    lines = text.expandtabs(4).splitlines() or [""]
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    char_width = probe.textlength("M", font=font)
    line_height = 21
    pad = 24
    width = int(max(len(line) for line in lines) * char_width) + pad * 2
    height = pad * 2 + len(lines) * line_height + 44
    image = Image.new("RGB", (max(width, 780), height), BG)
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, image.width, 34], fill=CHROME)
    for index, color in enumerate(("#ff5f56", "#ffbd2e", "#27c93f")):
        draw.ellipse([16 + index * 22, 10, 28 + index * 22, 22], fill=color)
    draw.text((92, 9), title, font=load_font(13), fill=MUTED)

    for row, line in enumerate(lines):
        color = FG
        stripped = line.strip()
        if line.startswith("$"):
            color = ACCENT
        elif stripped.startswith(("tokenizer", "==", "vocab")) or "Tokenizers" in stripped:
            color = MUTED
        elif stripped.startswith("bpe-"):
            color = "#1a7f37"
        draw.text((pad, 50 + row * line_height), line, font=font, fill=color)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def render_fertility(rows: list[dict], chars: float, words: float, out_path: Path) -> None:
    vocab = [row["vocab"] for row in rows]
    fertility = [row["tokens_per_word"] for row in rows]

    figure, axis = plt.subplots(figsize=(9.4, 4.0))
    figure.patch.set_facecolor("#ffffff")
    axis.plot(vocab, fertility, marker="o", color="#0969da", label="BPE (learned merges)")
    axis.axhline(words, color="#1a7f37", linestyle="--", linewidth=1.2,
                 label=f"word-level = {words:.2f} tokens/word")
    axis.axhline(chars, color="#cf222e", linestyle="--", linewidth=1.2,
                 label=f"char-level = {chars:.2f} tokens/word")
    for row in rows:
        axis.annotate(
            f"{row['tokens_per_word']:.2f}",
            (row["vocab"], row["tokens_per_word"]),
            textcoords="offset points",
            xytext=(0, 8),
            fontsize=7,
            ha="center",
            color="#0969da",
        )
    axis.set_xlabel("BPE vocabulary size (symbols)")
    axis.set_ylabel("tokens per word on held-out text")
    axis.set_title("Most of the win arrives early; the last merges buy almost nothing", fontsize=10,
                   loc="left")
    axis.grid(linestyle=":", alpha=0.45)
    axis.legend(fontsize=8, frameon=False)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    figure.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, format=out_path.suffix.lstrip("."))
    plt.close(figure)


def render_segmentation(corpus: str, out_path: Path) -> None:
    models = [(size, train(corpus, size)) for size in (300, 600, 883)]
    figure, axis = plt.subplots(figsize=(11.0, 0.85 * len(models) + 1.6))
    figure.patch.set_facecolor("#ffffff")
    palette = ["#dbeafe", "#bfdbfe", "#93c5fd", "#60a5fa", "#3b82f6"]
    widest = 0.0

    for row, (size, model) in enumerate(models):  # noqa: B007 - size is informational
        tokens = encode(SENTENCE, model)
        x = 0.0
        for index, token in enumerate(tokens):
            width = len(token.replace(END_OF_WORD, "")) + 0.35
            axis.add_patch(
                Rectangle((x, row - 0.32), width, 0.64,
                          facecolor=palette[index % len(palette)], edgecolor="#1f6feb",
                          linewidth=0.6)
            )
            label = token.replace(END_OF_WORD, "·")
            axis.text(x + width / 2, row, label, ha="center", va="center", fontsize=6.5,
                      color="#0b1f33")
            x += width + 0.08
        widest = max(widest, x)
        axis.text(-0.6, row, f"vocab {model.size}\n{len(tokens)} tokens", ha="right",
                  va="center", fontsize=8)

    axis.set_xlim(-9.5, widest + 0.5)
    axis.set_ylim(len(models) - 0.5, -0.9)
    axis.axis("off")
    axis.set_title(
        "The same sentence at three vocabulary sizes "
        "(· = end of word; wide boxes are full words, narrow ones are character fallbacks)",
        fontsize=9,
        loc="left",
    )
    figure.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, format="png", dpi=150)
    plt.close(figure)


def main() -> int:
    images = ROOT / "docs" / "images"
    images.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)

    corpus = Path("examples/corpus.txt").read_text(encoding="utf-8")
    heldout = Path("examples/heldout.txt").read_text(encoding="utf-8")

    rows = []
    for size in VOCABS:
        model = train(corpus, size)
        score = score_bpe(f"bpe-{model.size}", heldout, model)
        rows.append(
            {
                **score.as_dict(),
                "vocab": model.size,
                "merges_fired": len(model.merges) - unused_merges(model, heldout),
            }
        )

    mid = train(corpus, 600)
    bpe = score_bpe(f"bpe-{mid.size}", heldout, mid)
    chars = score_chars(heldout)
    word_level = score_words(heldout)

    payload = {
        "corpus_chars": len(corpus),
        "heldout_words": chars.words,
        "sweep": rows,
        "comparison": [chars.as_dict(), word_level.as_dict(), bpe.as_dict()],
        "sample_tokens": encode(SENTENCE, train(corpus, 883)),
    }
    (ROOT / "docs" / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    terminal = (
        "$ subword-probe compare --vocab 600\n\n"
        + render_table([chars, word_level, bpe])
        + "\n\n"
        + render_score(bpe)
        + "\n\n$ subword-probe sweep\n\n"
        + render_sweep(rows)
    )
    title = "subword-probe  -  a tokenizer you can read in one sitting"
    render_terminal(terminal, images / "compare.png", title)
    render_fertility(rows, chars.tokens_per_word, word_level.tokens_per_word,
                     images / "fertility-curve.svg")
    render_segmentation(corpus, images / "segmentation.png")

    print(
        " | ".join(f"{row['vocab']}->{row['tokens_per_word']:.2f}" for row in rows)
        + f" | char {chars.tokens_per_word:.2f} word {word_level.tokens_per_word:.2f}"
        + f" | bpe600 used {bpe.vocab_used}/{bpe.vocab_size}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
