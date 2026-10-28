"""A byte-pair-encoding tokenizer in about a hundred lines, so every number is inspectable.

Vocabulary starts with the characters that appear in the corpus plus an end-of-word marker, then
grows by merging the most frequent adjacent pair, one symbol per merge. Encoding replays the merges
in the order they were learned, which is the whole algorithm.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

END_OF_WORD = "</w>"


class BpeError(ValueError):
    """Raised when a corpus or model cannot be used."""


@dataclass(frozen=True)
class Model:
    merges: tuple[tuple[str, str], ...]
    base: tuple[str, ...]

    @property
    def vocab(self) -> tuple[str, ...]:
        return self.base + tuple(left + right for left, right in self.merges)

    @property
    def size(self) -> int:
        return len(self.vocab)

    def rank(self) -> dict[tuple[str, str], int]:
        return {pair: index for index, pair in enumerate(self.merges)}

    def as_dict(self) -> dict:
        return {
            "size": self.size,
            "base": list(self.base),
            "merges": [list(pair) for pair in self.merges],
        }


def words(text: str) -> Counter[tuple[str, ...]]:
    """Split on whitespace and represent every word as characters plus an end marker."""
    counter: Counter[tuple[str, ...]] = Counter()
    for raw in text.split():
        if raw:
            counter[tuple(raw) + (END_OF_WORD,)] += 1
    return counter


def _count_pairs(corpus: Counter[tuple[str, ...]]) -> Counter[tuple[str, str]]:
    pairs: Counter[tuple[str, str]] = Counter()
    for symbols, frequency in corpus.items():
        for left, right in zip(symbols, symbols[1:], strict=False):
            pairs[(left, right)] += frequency
    return pairs


def _merge_symbols(symbols: tuple[str, ...], pair: tuple[str, str]) -> tuple[str, ...]:
    merged: list[str] = []
    index = 0
    while index < len(symbols):
        if index + 1 < len(symbols) and (symbols[index], symbols[index + 1]) == pair:
            merged.append(symbols[index] + symbols[index + 1])
            index += 2
        else:
            merged.append(symbols[index])
            index += 1
    return tuple(merged)


def train(text: str, vocab_size: int) -> Model:
    """Learn merges until the vocabulary reaches ``vocab_size`` symbols."""
    corpus = words(text)
    if not corpus:
        raise BpeError("corpus is empty")
    base = tuple(sorted({symbol for word in corpus for symbol in word}))
    if vocab_size <= len(base):
        raise BpeError(
            f"vocab_size {vocab_size} must exceed the {len(base)} base symbols in this corpus"
        )

    merges: list[tuple[str, str]] = []
    working = Counter(corpus)
    while len(base) + len(merges) < vocab_size:
        pairs = _count_pairs(working)
        if not pairs:
            break
        # Ties break on the pair itself so training is reproducible across runs.
        pair = max(pairs.items(), key=lambda item: (item[1], item[0]))[0]
        merges.append(pair)
        working = Counter(
            {_merge_symbols(symbols, pair): frequency for symbols, frequency in working.items()}
        )
    return Model(merges=tuple(merges), base=base)


def encode_word(
    word: str, model: Model, ranks: dict[tuple[str, str], int] | None = None
) -> list[str]:
    ranks = ranks if ranks is not None else model.rank()
    symbols = list(word) + [END_OF_WORD]
    while len(symbols) > 1:
        candidates = [
            (ranks[(symbols[index], symbols[index + 1])], index)
            for index in range(len(symbols) - 1)
            if (symbols[index], symbols[index + 1]) in ranks
        ]
        if not candidates:
            break
        _rank, position = min(candidates)
        symbols[position : position + 2] = [symbols[position] + symbols[position + 1]]
    return symbols


def encode(text: str, model: Model) -> list[str]:
    ranks = model.rank()
    tokens: list[str] = []
    for raw in text.split():
        if raw:
            tokens.extend(encode_word(raw, model, ranks))
    return tokens


def decode(tokens: list[str]) -> str:
    """Invert :func:`encode` - every token sequence maps back to the original text."""
    joined = "".join(tokens)
    parts = [part for part in joined.split(END_OF_WORD) if part]
    return " ".join(parts)
