"""Tests for the paired statistics helpers."""
import pytest

pytest.importorskip("scipy")

from core.experiment import stats  # noqa: E402


def test_a12_known_values():
    assert stats.a12([1, 2, 3], [1, 2, 3]) == 0.5
    assert stats.a12([4, 5], [1, 2]) == 1.0
    assert stats.a12([1, 2], [2, 3]) == 0.125


def test_holm_known_values():
    adjusted = stats.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted["a"] == pytest.approx(0.03)
    assert adjusted["b"] == pytest.approx(0.06)
    assert adjusted["c"] == pytest.approx(0.06)


def test_holm_is_capped_at_one():
    assert stats.holm({"a": 0.9, "b": 0.8})["a"] == 1.0


def test_mcnemar_no_discordant_pairs_is_one():
    assert stats.mcnemar_exact([True, False, True], [True, False, True]) == 1.0


def test_mcnemar_known_value():
    from scipy.stats import binomtest
    a = [True] * 8 + [True, False]
    b = [False] * 8 + [True, False]
    assert stats.mcnemar_exact(a, b) == pytest.approx(binomtest(8, 8).pvalue)


def test_wilcoxon_all_equal_is_one():
    assert stats.wilcoxon_paired([1, 2, 3], [1, 2, 3]) == 1.0


def test_wilcoxon_too_few_pairs_is_none():
    assert stats.wilcoxon_paired([1], [2]) is None


def test_wilcoxon_detects_shift():
    p = stats.wilcoxon_paired([1, 2, 3, 4, 5, 6, 7, 8], [11, 12, 13, 14, 15, 16, 17, 18])
    assert p is not None and p < 0.01
