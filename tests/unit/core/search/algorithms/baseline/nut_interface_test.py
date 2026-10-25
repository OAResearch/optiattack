"""Tests for the baseline NUT interface helpers."""
import numpy as np
import pytest

from core.search.algorithms.baseline.nut_interface import (
    InvalidScoresError, NutCapabilityError, NutRequirement, check_requirement, to_one_hot, to_probability_vector)
from core.utils.nut_request import NutRequest


def _result(**scores):
    return NutRequest({"predictions": [{"label": k, "score": v} for k, v in scores.items()]})


LABELS = ["a", "b", "c", "d"]


def test_vector_spreads_remaining_mass_over_unreported_labels():
    vector = to_probability_vector(_result(a=0.6, b=0.2), LABELS)
    assert np.allclose(vector, [0.6, 0.2, 0.1, 0.1])


def test_vector_with_all_labels_reported_is_returned_as_is():
    vector = to_probability_vector(_result(a=0.5, b=0.3), ["a", "b"])
    assert np.allclose(vector, [0.5, 0.3])


def test_unknown_label_raises():
    with pytest.raises(InvalidScoresError, match="zzz"):
        to_probability_vector(_result(a=0.6, zzz=0.2), LABELS)


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_negative_or_above_one_score_raises(bad):
    with pytest.raises(InvalidScoresError, match=str(bad)):
        to_probability_vector(_result(a=bad, b=0.2), LABELS)


def test_scores_summing_above_one_raise():
    with pytest.raises(InvalidScoresError):
        to_probability_vector(_result(a=0.8, b=0.4), LABELS)


def test_one_hot_marks_argmax():
    assert to_one_hot(np.array([0.1, 0.7, 0.2])).tolist() == [0.0, 1.0, 0.0]


def test_missing_labels_raise_helpful_error():
    for info in ({}, {"labels": None}, {"labels": []}):
        with pytest.raises(NutCapabilityError, match="labels="):
            check_requirement(NutRequirement.SCORES, info)


def test_gradients_requirement_rejected_for_scores_nut():
    with pytest.raises(NutCapabilityError):
        check_requirement(NutRequirement.GRADIENTS, {"labels": ["a", "b"], "capabilities": ["scores"]})


def test_label_requirement_accepted_for_scores_nut():
    info = {"labels": ["a", "b"], "capabilities": ["scores"]}
    assert check_requirement(NutRequirement.LABEL, info) == ["a", "b"]
    assert check_requirement(NutRequirement.SCORES, info) == ["a", "b"]


def test_duplicate_labels_in_class_list_raise():
    with pytest.raises(NutCapabilityError, match="crane") as error:
        check_requirement(NutRequirement.SCORES, {"labels": ["crane", "dog", "crane", "cat"]})
    assert "unique" in str(error.value)
    assert "dog" not in str(error.value).split("unique")[0]


def test_label_reported_twice_in_one_result_raises():
    result = NutRequest({"predictions": [{"label": "a", "score": 0.3}, {"label": "b", "score": 0.1},
                                         {"label": "a", "score": 0.2}]})
    with pytest.raises(InvalidScoresError, match="'a'"):
        to_probability_vector(result, LABELS)
