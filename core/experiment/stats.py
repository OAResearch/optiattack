"""Paired statistics for comparing attack methods: Wilcoxon, Vargha-Delaney A12, McNemar and Holm."""
from typing import Any, Dict, Optional, Sequence

INSTALL_HINT = "Experiment statistics need scipy. Install it with: pip install -r requirements-experiments.txt"


def _scipy_stats() -> Any:
    """Import scipy.stats lazily."""
    try:
        from scipy import stats  # type: ignore[import-untyped,unused-ignore]
    except ImportError as e:
        raise ImportError(INSTALL_HINT) from e
    return stats


def a12(x: Sequence[float], y: Sequence[float]) -> float:
    """Vargha-Delaney A12: the probability that a value of x is larger than one of y, plus half of the ties."""
    if not x or not y:
        raise ValueError("a12 needs two non-empty samples")
    score = 0.0
    for a in x:
        for b in y:
            score += 1.0 if a > b else 0.5 if a == b else 0.0
    return score / (len(x) * len(y))


def wilcoxon_paired(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    """Paired Wilcoxon signed-rank p-value; 1.0 if all differences are zero, None for fewer than 2 pairs."""
    if len(x) != len(y):
        raise ValueError("wilcoxon_paired needs samples of equal length")
    if len(x) < 2:
        return None
    if all(a == b for a, b in zip(x, y)):
        return 1.0
    return float(_scipy_stats().wilcoxon(x, y).pvalue)


def mcnemar_exact(a_success: Sequence[bool], b_success: Sequence[bool]) -> float:
    """Exact two-sided McNemar test on the discordant pairs; 1.0 if there are none."""
    if len(a_success) != len(b_success):
        raise ValueError("mcnemar_exact needs samples of equal length")
    a_only = sum(1 for a, b in zip(a_success, b_success) if a and not b)
    b_only = sum(1 for a, b in zip(a_success, b_success) if b and not a)
    discordant = a_only + b_only
    if discordant == 0:
        return 1.0
    return float(_scipy_stats().binomtest(a_only, discordant, 0.5).pvalue)


def holm(pvalues: Dict[str, float]) -> Dict[str, float]:
    """Holm step-down adjusted p-values: monotone and capped at 1."""
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    m = len(ordered)
    adjusted: Dict[str, float] = {}
    running = 0.0
    for rank, (key, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[key] = running
    return adjusted
