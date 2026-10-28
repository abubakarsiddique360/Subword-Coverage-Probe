"""Coverage metrics: what a tokenizer costs on a corpus, and how much vocabulary it wastes."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .bpe import Model, encode


@dataclass(frozen=True)
class Score:
    name: str
    tokens: int
    words: int
    chars: int
    vocab_size: int
    vocab_used: int

    @property
    def tokens_per_word(self) -> float:
        return self.tokens / self.words if self.words else 0.0

    @property
    def chars_per_token(self) -> float:
        return self.chars / self.tokens if self.tokens else 0.0

    @property
    def utilisation(self) -> float:
        return self.vocab_used / self.vocab_size if self.vocab_size else 0.0

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "tokens": self.tokens,
            "words": self.words,
            "chars": self.chars,
            "vocab_size": self.vocab_size,
            "vocab_used": self.vocab_used,
            "tokens_per_word": round(self.tokens_per_word, 4),
            "chars_per_token": round(self.chars_per_token, 4),
            "utilisation": round(self.utilisation, 4),
        }


def _counts(text: str) -> tuple[int, int]:
    tokens = text.split()
    return len(tokens), sum(len(token) for token in tokens)


def score_bpe(name: str, text: str, model: Model) -> Score:
    word_count, char_count = _counts(text)
    tokens = encode(text, model)
    return Score(
        name=name,
        tokens=len(tokens),
        words=word_count,
        chars=char_count,
        vocab_size=model.size,
        vocab_used=len(set(tokens)),
    )


def score_chars(text: str) -> Score:
    """Character tokenizer: one token per character, vocabulary is the alphabet."""
    word_count, char_count = _counts(text)
    alphabet = {char for char in text if not char.isspace()}
    return Score(
        name="char-level",
        tokens=char_count,
        words=word_count,
        chars=char_count,
        vocab_size=len(alphabet),
        vocab_used=len(alphabet),
    )


def score_words(text: str) -> Score:
    """Whitespace tokenizer: one token per word, but the vocabulary is the whole lexicon."""
    word_count, char_count = _counts(text)
    lexicon = set(text.split())
    return Score(
        name="word-level",
        tokens=word_count,
        words=word_count,
        chars=char_count,
        vocab_size=len(lexicon),
        vocab_used=len(lexicon),
    )


def unused_merges(model: Model, text: str) -> int:
    """Learned merges that never fire on this text - vocabulary paid for and not used."""
    used = set(encode(text, model))
    return sum(1 for left, right in model.merges if left + right not in used)


def most_common_tokens(text: str, model: Model, limit: int = 8) -> list[tuple[str, int]]:
    counter: Counter[str] = Counter(encode(text, model))
    return counter.most_common(limit)


def oov_rate(model: Model, words: list[str]) -> float:
    """Share of words containing a character the tokenizer never saw while training.

    A character-level model is only unknown-free for characters in its base vocabulary. Anything
    outside it passes through as a single symbol - still losslessly decodable, but expensive (one
    token per character) and usually a sign the training corpus was too narrow.
    """
    if not words:
        return 0.0
    alphabet = set(model.base)
    unknown = sum(1 for word in words if any(char not in alphabet for char in word))
    return unknown / len(words)
