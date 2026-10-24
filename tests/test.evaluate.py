"""Unit tests for the CER / WER calculation in eval/evaluate.py."""
import pytest

from eval.evaluate import compute_cer_wer, guess_language, normalize_text


def test_known_pair_cer_and_wer():
    # one wrong character out of three; one wrong word out of three
    cer, _ = compute_cer_wer("abc", "abd")
    assert cer == pytest.approx(1 / 3)
    _, wer = compute_cer_wer("the cat sat", "the cat sit")
    assert wer == pytest.approx(1 / 3)


def test_hello_hallo():
    cer, wer = compute_cer_wer("hello", "hallo")
    assert cer == pytest.approx(0.2)
    assert wer == pytest.approx(1.0)


def test_deleted_words():
    _, wer = compute_cer_wer("a b c d", "a b")
    assert wer == pytest.approx(0.5)


def test_identical_text_is_zero():
    text = "আমি ভাত খাই"
    assert compute_cer_wer(text, text) == (0.0, 0.0)


def test_whitespace_differences_are_ignored():
    assert compute_cer_wer("a b\nc", "  a   b c ") == (0.0, 0.0)


def test_bengali_word_error():
    cer, wer = compute_cer_wer("আমি ভাত খাই", "আমি ভাল খাই")
    assert wer == pytest.approx(1 / 3)
    assert 0 < cer < 0.5


def test_unicode_nfc_normalisation():
    decomposed = "é"      # e + combining acute
    composed = "é"         # precomposed e-acute
    assert normalize_text(decomposed) == composed
    assert compute_cer_wer(composed, decomposed) == (0.0, 0.0)


def test_lowercase_is_optional():
    assert compute_cer_wer("Hello World", "hello world")[0] > 0
    assert compute_cer_wer("Hello World", "hello world", lowercase=True) == (0.0, 0.0)


def test_empty_reference():
    assert compute_cer_wer("", "") == (0.0, 0.0)
    assert compute_cer_wer("", "something") == (1.0, 1.0)


def test_guess_language_from_prefix():
    assert guess_language("bn_01.png") == "bn"
    assert guess_language("HI_02_deg.png") == "hi"
    assert guess_language("en_03.jpg") == "en"
    assert guess_language("scan.png") == "unknown"
