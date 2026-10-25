"""Declares what an attack needs from the NUT and converts NUT results to probability vectors."""
from enum import Enum
from typing import List

import numpy as np

from core.utils.nut_request import NutRequest


class NutRequirement(Enum):

    """Information an attack needs from the NUT."""

    SCORES = "scores"
    LABEL = "label"
    GRADIENTS = "gradients"


class NutCapabilityError(ValueError):

    """Raised when the NUT cannot provide what an attack requires."""


class InvalidScoresError(ValueError):

    """Raised when the scores returned by the NUT are not usable."""


_PROVIDED_BY = {
    NutRequirement.SCORES: {"scores"},
    NutRequirement.LABEL: {"scores", "label"},
    NutRequirement.GRADIENTS: {"gradients"},
}


def check_requirement(requirement: NutRequirement, nut_info: dict) -> List[str]:
    """Checks the NUT info against a requirement and returns the NUT class list."""
    labels = nut_info.get("labels")
    if not labels:
        raise NutCapabilityError(
            "The NUT did not report its class list ('labels'). Pass it from the client, e.g. "
            "client.collect_info(..., labels=['cat', 'dog', ...])."
        )
    duplicates = sorted({str(label) for label in labels if list(labels).count(label) > 1})
    if duplicates:
        raise NutCapabilityError(
            "The NUT class list ('labels') contains duplicated names %s. Class names must be unique; use e.g. "
            "WordNet synset IDs (n01440764, ...) instead of ambiguous names." % duplicates
        )
    capabilities = set(nut_info.get("capabilities") or ["scores"])
    if not capabilities & _PROVIDED_BY[requirement]:
        raise NutCapabilityError(
            "The attack requires '%s' but the NUT only provides %s." % (requirement.value, sorted(capabilities))
        )
    return list(labels)


def to_probability_vector(result: NutRequest, labels: List[str], tolerance: float = 1e-6) -> np.ndarray:
    """Builds a full probability vector, spreading the remaining mass over unreported labels."""
    index = {label: i for i, label in enumerate(labels)}
    vector = np.zeros(len(labels), dtype=float)
    total = 0.0
    seen = set()
    for prediction in result.predictions:
        if prediction.label not in index:
            raise InvalidScoresError("Unknown label '%s' returned by the NUT." % prediction.label)
        if prediction.label in seen:
            raise InvalidScoresError("Label '%s' is returned more than once by the NUT." % prediction.label)
        seen.add(prediction.label)
        score = float(prediction.value)
        if not 0.0 <= score <= 1.0:
            raise InvalidScoresError("Score %s for label '%s' is outside [0, 1]." % (score, prediction.label))
        vector[index[prediction.label]] = score
        total += score
    if total > 1.0 + tolerance:
        raise InvalidScoresError("Scores sum to %s which exceeds 1." % total)
    unreported = len(labels) - len(seen)
    if unreported > 0:
        remaining = max(0.0, 1.0 - total) / unreported
        reported = {index[label] for label in seen}
        for i in range(len(labels)):
            if i not in reported:
                vector[i] = remaining
    return vector


def to_one_hot(vector: np.ndarray) -> np.ndarray:
    """Returns a vector with 1 at the argmax and 0 elsewhere."""
    one_hot = np.zeros_like(vector, dtype=float)
    one_hot[int(np.argmax(vector))] = 1.0
    return one_hot
