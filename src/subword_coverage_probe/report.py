"""Text reports for tokenizer coverage."""

from __future__ import annotations

import json

from .bpe import Model
from .coverage import Score


def render_score(score: Score) -> str:
    lines = [f"Coverage: {score.name}", "=" * 58]
    lines.append(f"tokens            {score.tokens}")
    lines.append(f"words             {score.words}")
    lines.append(f"characters        {score.chars}")
    lines.append(f"tokens per word   {score.tokens_per_word:.2f}")
    lines.append(f"chars per token   {score.chars_per_token:.2f}")
    lines.append(f"vocabulary        {score.vocab_size} ({score.vocab_used} used, "
                 f"{score.utilisation * 100:.0f}%)")
    return "\n".join(lines)


def render_table(scores: list[Score]) -> str:
    header = (
        f"{'tokenizer':<18}{'vocab':>7}{'used':>7}{'tokens':>8}"
        f"{'tok/word':>10}{'chars/tok':>11}"
    )
    lines = ["Tokenizers on the same text", "=" * len(header), header]
    for score in scores:
        lines.append(
            f"{score.name[:17]:<18}{score.vocab_size:>7}{score.vocab_used:>7}"
            f"{score.tokens:>8}{score.tokens_per_word:>10.2f}{score.chars_per_token:>11.2f}"
        )
    return "\n".join(lines)


def render_sweep(rows: list[dict]) -> str:
    header = f"{'vocab':>7}{'tokens':>9}{'tok/word':>10}{'chars/tok':>11}{'merges fired':>14}"
    lines = ["Vocabulary size against cost", "=" * len(header), header]
    for row in rows:
        lines.append(
            f"{row['vocab']:>7}{row['tokens']:>9}{row['tokens_per_word']:>10.2f}"
            f"{row['chars_per_token']:>11.2f}{row['merges_fired']:>14}"
        )
    lines.append("")
    lines.append("merges fired = learned merges that appear in the text at least once")
    return "\n".join(lines)


def render_merges(model: Model, limit: int = 20) -> str:
    lines = [f"Learned merges ({len(model.merges)} total, first {min(limit, len(model.merges))})",
             "=" * 58]
    for index, (left, right) in enumerate(model.merges[:limit], start=1):
        lines.append(f"{index:>3}. {left} + {right} -> {left}{right}")
    lines.append("")
    lines.append(f"base symbols: {len(model.base)}")
    return "\n".join(lines)


def render_tokens(tokens: list[str], limit: int = 60) -> str:
    shown = tokens[:limit]
    text = " | ".join(shown)
    if len(tokens) > limit:
        text += f" … (+{len(tokens) - limit} more)"
    return text


def render_json(payload) -> str:
    return json.dumps(payload, indent=2)
