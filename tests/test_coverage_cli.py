import json

import pytest

from subword_coverage_probe.bpe import train
from subword_coverage_probe.cli import main
from subword_coverage_probe.coverage import (
    most_common_tokens,
    oov_rate,
    score_bpe,
    score_chars,
    score_words,
    unused_merges,
)
from subword_coverage_probe.report import (
    render_json,
    render_merges,
    render_score,
    render_sweep,
    render_table,
    render_tokens,
)

CORPUS = "the lower the lower the lowest lower newer newer newest new the lower"
HELDOUT = "the lower newer lowest unseen"


def model():
    return train(CORPUS, 120)


def test_char_score_uses_alphabet_as_vocab():
    score = score_chars("ab ab")
    assert score.tokens == 4
    assert score.words == 2
    assert score.vocab_size == 2
    assert score.tokens_per_word == 2.0


def test_word_score_has_one_token_per_word():
    score = score_words(HELDOUT)
    assert score.tokens == score.words
    assert score.tokens_per_word == 1.0
    assert score.vocab_size == 5


def test_bpe_score_beats_characters():
    bpe = score_bpe("bpe", HELDOUT, model())
    chars = score_chars(HELDOUT)
    assert bpe.tokens < chars.tokens
    assert bpe.vocab_size > chars.vocab_size


def test_utilisation_is_used_over_total():
    score = score_bpe("bpe", HELDOUT, model())
    assert 0 < score.utilisation <= 1


def test_unused_merges_on_tiny_text():
    big = train(CORPUS, 130)
    assert unused_merges(big, "the") >= 0


def test_most_common_tokens_are_sorted():
    counts = most_common_tokens(CORPUS, model(), 3)
    values = [count for _token, count in counts]
    assert values == sorted(values, reverse=True)


def test_oov_is_zero_for_known_characters():
    assert oov_rate(model(), ["the", "lower", "newest"]) == 0.0


def test_oov_flags_characters_unseen_in_training():
    # the tiny test corpus contains no 'z', so any word with one is "unknown" at character level
    assert oov_rate(model(), ["zzz"]) == 1.0


def test_oov_of_empty_list_is_zero():
    assert oov_rate(model(), []) == 0.0


def test_report_renders():
    score = score_bpe("bpe", HELDOUT, model())
    assert "tokens per word" in render_score(score)
    assert "tok/word" in render_table([score_chars(HELDOUT), score_words(HELDOUT), score])
    assert "Learned merges" in render_merges(model(), 3)
    assert " | " in render_tokens(["a", "b"])
    rows = [{"vocab": 120, "tokens": 10, "tokens_per_word": 1.5, "chars_per_token": 2.0,
             "merges_fired": 7}]
    assert "merges fired" in render_sweep(rows)


def test_json_report():
    payload = json.loads(render_json([score_bpe("bpe", HELDOUT, model()).as_dict()]))
    assert payload[0]["name"] == "bpe"


def test_render_tokens_truncates():
    text = render_tokens([f"t{index}" for index in range(70)], limit=5)
    assert "more" in text


@pytest.fixture
def files(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(CORPUS, encoding="utf-8")
    heldout = tmp_path / "heldout.txt"
    heldout.write_text(HELDOUT, encoding="utf-8")
    return corpus, heldout


def test_cli_train(files, capsys):
    corpus, _ = files
    assert main(["train", "--corpus", str(corpus), "--vocab", "120"]) == 0
    output = capsys.readouterr().out
    assert "Learned merges" in output
    assert "unused merges" in output


def test_cli_train_bad_vocab(files, capsys):
    corpus, _ = files
    assert main(["train", "--corpus", str(corpus), "--vocab", "2"]) == 2
    assert "must exceed" in capsys.readouterr().err


def test_cli_train_missing_corpus(tmp_path, capsys):
    assert main(["train", "--corpus", str(tmp_path / "nope.txt")]) == 2
    assert "cannot read" in capsys.readouterr().err


def test_cli_encode(files, capsys):
    corpus, _ = files
    assert main(["encode", "--corpus", str(corpus), "--vocab", "120", "--text", "the lower"]) == 0
    output = capsys.readouterr().out
    assert "round trip: 'the lower'" in output


def test_cli_compare(files, capsys):
    corpus, heldout = files
    argv = ["compare", "--corpus", str(corpus), "--heldout", str(heldout), "--vocab", "120"]
    assert main(argv) == 0
    output = capsys.readouterr().out
    assert "char-level" in output
    assert "word-level" in output
    assert "oov rate" in output


def test_cli_compare_json(files, capsys):
    corpus, heldout = files
    argv = ["compare", "--corpus", str(corpus), "--heldout", str(heldout), "--vocab", "120",
            "--format", "json"]
    assert main(argv) == 0
    assert len(json.loads(capsys.readouterr().out)) == 3


def test_cli_sweep(files, capsys):
    corpus, heldout = files
    argv = ["sweep", "--corpus", str(corpus), "--heldout", str(heldout), "--vocabs", "110,120"]
    assert main(argv) == 0
    assert "Vocabulary size against cost" in capsys.readouterr().out


def test_cli_sweep_notes_saturation(files, capsys):
    corpus, heldout = files
    argv = ["sweep", "--corpus", str(corpus), "--heldout", str(heldout), "--vocabs", "5000"]
    assert main(argv) == 0
    assert "no adjacent pairs left" in capsys.readouterr().out


def test_cli_common(files, capsys):
    corpus, _ = files
    assert main(["common", "--corpus", str(corpus), "--vocab", "120"]) == 0
    assert "the" in capsys.readouterr().out
