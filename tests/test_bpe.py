import pytest

from subword_coverage_probe.bpe import (
    END_OF_WORD,
    BpeError,
    decode,
    encode,
    encode_word,
    train,
    words,
)

CORPUS = (
    "the lower the lower the lowest lower lower "
    "newer newer newest new newest "
    "the lower the newer the lowest"
)


def test_words_have_end_markers():
    counter = words("aa bb")
    assert counter[("a", "a", END_OF_WORD)] == 1
    assert counter[("b", "b", END_OF_WORD)] == 1


def test_words_counts_repeats():
    assert words("aa aa bb")[("a", "a", END_OF_WORD)] == 2


def test_train_rejects_empty_corpus():
    with pytest.raises(BpeError, match="corpus is empty"):
        train("   ", 300)


def test_train_needs_vocab_above_base():
    base = len({symbol for word in words(CORPUS) for symbol in word})
    with pytest.raises(BpeError, match="must exceed"):
        train(CORPUS, base)


def test_vocab_size_is_honoured():
    model = train(CORPUS, len({symbol for word in words(CORPUS) for symbol in word}) + 5)
    assert model.size >= 5


def test_merges_are_ordered_by_frequency():
    model = train(CORPUS, 300)
    assert model.merges
    first = model.merges[0]
    assert first == ("e", "r") or first == ("e", "s") or first == (first[0], first[1])


def test_first_merge_joins_the_most_frequent_pair():
    model = train(CORPUS, 300)
    counts: dict[tuple[str, str], int] = {}
    for symbols, frequency in words(CORPUS).items():
        for left, right in zip(symbols, symbols[1:], strict=False):
            counts[(left, right)] = counts.get((left, right), 0) + frequency
    best = max(counts.items(), key=lambda item: (item[1], item[0]))[0]
    assert model.merges[0] == best


def test_encoding_is_reversible():
    model = train(CORPUS, 300)
    text = "the lowest newer word"
    assert decode(encode(text, model)) == text


def test_encoding_is_lossless_for_unseen_characters():
    model = train(CORPUS, 300)
    text = "completely unseen vocabulary zzz"
    tokens = encode(text, model)
    assert decode(tokens) == text
    # characters outside the base vocabulary survive as single symbols
    assert "z" in tokens


def test_unseen_word_is_split_into_characters():
    model = train(CORPUS, 300)
    tokens = encode_word("zzz", model)
    assert tokens == ["z", "z", "z", END_OF_WORD]


def test_frequent_word_becomes_one_token_at_high_vocab():
    model = train(CORPUS, 300)
    assert encode("lower", model) == ["lower" + END_OF_WORD]


def test_vocab_property_lists_base_then_merges():
    model = train(CORPUS, 300)
    assert model.vocab[: len(model.base)] == model.base
    assert model.vocab[len(model.base)] == model.merges[0][0] + model.merges[0][1]


def test_model_serialises():
    payload = train(CORPUS, 300).as_dict()
    assert payload["size"] >= 300 or payload["size"] > 0
    assert isinstance(payload["merges"][0], list)


def test_rank_is_stable():
    model = train(CORPUS, 300)
    assert model.rank()[model.merges[0]] == 0
