"""Split-conformal p-values (Vovk, Gammerman & Shafer 2005).

A conformal p-value is the only *distribution-free* way we know of to turn an
arbitrary black-box score into a calibrated uncertainty statement: under
exchangeability alone it satisfies

    P( p <= u ) <= u      for every u in [0, 1]        (super-uniformity)

with no assumption whatsoever on the score distribution, the model, or the
answer alphabet.  That property is exactly the input the Vovk-Wang p-to-e
calibrator needs in order to emit a *valid e-variable*, which is what lets us
fold arbitrary verifiers into an anytime-valid e-process (see ``verifier.py``).

Author: 晨星 (CJX0712)
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = ["conformal_pvalue", "conformal_pvalue_batch", "aggregate_scores"]


def conformal_pvalue(
    score: float,
    calibration: Sequence[float] | np.ndarray,
    higher_is_more_extreme: bool = True,
) -> float:
    """Rank-based conformal p-value of ``score`` against a calibration set.

    ``p = ( 1 + #{c in calib : c >= score} ) / (n + 1)``   (higher = more extreme)
    ``p = ( 1 + #{c in calib : c <= score} ) / (n + 1)``   (lower  = more extreme)

    Under exchangeability of ``score`` with the calibration scores this is
    super-uniform.  The ``+1`` in both numerator and denominator is the
    *smoothed* (a.k.a. randomised-free) convention; it keeps ``p >= 1/(n+1)``
    strictly positive, which in turn keeps the derived e-value finite.

    Parameters
    ----------
    score:
        Observed score for the candidate being tested.
    calibration:
        Scores drawn from the *null* (reference) population.
    higher_is_more_extreme:
        Direction of the deviation that counts as evidence.
    """
    cal = np.asarray(calibration, dtype=np.float64).ravel()
    n = cal.size
    if n == 0:
        raise ValueError("calibration set must be non-empty")
    s = float(score)
    if higher_is_more_extreme:
        cnt = int(np.sum(cal >= s))
    else:
        cnt = int(np.sum(cal <= s))
    return (1.0 + cnt) / (n + 1.0)


def conformal_pvalue_batch(
    scores: Sequence[float] | np.ndarray,
    calibration: Sequence[float] | np.ndarray,
    higher_is_more_extreme: bool = True,
) -> np.ndarray:
    """Vectorised :func:`conformal_pvalue`."""
    cal = np.asarray(calibration, dtype=np.float64).ravel()
    n = cal.size
    if n == 0:
        raise ValueError("calibration set must be non-empty")
    s = np.asarray(scores, dtype=np.float64).ravel()
    if higher_is_more_extreme:
        cnt = (cal[None, :] >= s[:, None]).sum(axis=1)
    else:
        cnt = (cal[None, :] <= s[:, None]).sum(axis=1)
    return (1.0 + cnt) / (n + 1.0)


def aggregate_scores(scores: Sequence[float] | np.ndarray, rule: str = "mean") -> float:
    """Collapse the per-trajectory scores of one candidate into a single score.

    ``mean`` is the robust default (it is the natural plug-in for the expected
    verifier score); ``max`` is sharper when at least one trajectory is likely
    to be a clean deduction but inflates variance; ``logmeanexp`` interpolates.
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    if s.size == 0:
        return float("nan")
    if rule == "mean":
        return float(np.mean(s))
    if rule == "max":
        return float(np.max(s))
    if rule == "logmeanexp":
        m = float(np.max(s))
        return m + float(np.log(np.mean(np.exp(s - m))))
    raise ValueError(f"unknown aggregation rule: {rule!r}")
