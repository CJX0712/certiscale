"""Verifier-Calibrated E-process (VCE) -- CertiScale's first novel component.

The gap this closes
-------------------
CITE (arXiv:2605.05873) decides *"is r the modal answer?"*.  It never looks at
**whether r is right** -- a model can be confidently and consistently wrong.  In
practice every serious LLM deployment has cheap auxiliary verifiers (a unit
test, a symbolic checker, a retrieval overlap score, a small reward model), and
throwing that signal away is pure waste.

The difficulty is that a verifier score is **not** a p-value and **not** an
e-value: naively exponentiating it, or normalising it into [0, 1], destroys any
error guarantee and re-introduces the very peeking problem CITE was built to
solve.  VCE solves this in two provably-valid steps:

    1. Split-conformal rank  ->  super-uniform p-value
       (Vovk/Gammerman/Shafer 2005; valid under exchangeability alone)
    2. Vovk-Wang calibrator  ->  e-value with E[e] <= 1 under the null
       (Vovk & Wang 2021, Annals of Statistics 49(3))

The resulting e-values are then combined **by arithmetic mean**, which is
validity-preserving with *no independence assumption* -- essential, because
real verifiers (an NLI model and a retrieval score, say) are strongly
correlated and a product rule would silently invalidate the guarantee.

Multiple verifiers are therefore fused *conservatively but safely*, and the
whole bank is folded into the CITE stopping rule through a union bound
(see ``flagship.py``), so the anytime-valid Type-I error is preserved exactly.

Author: 晨星 (CJX0712)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .conformal import aggregate_scores, conformal_pvalue
from .eprocess import (
    combine_mean,
    vovk_wang_calibrator,
)

__all__ = ["ConformalEVerifier", "VerifierBank", "vce_eprocess"]


@dataclass(frozen=True)
class ConformalEVerifier:
    """One verifier, calibrated into a valid e-value emitter.

    Parameters
    ----------
    name:
        Human-readable identifier (used in audit trails).
    null_calibration:
        Verifier scores on **reference negatives** -- historically wrong
        candidates (or, more generally, any population whose exchangeability
        with the null hypothesis we are willing to assume).
    kappa:
        Vovk-Wang calibrator exponent in ``(0, 1)``.  Smaller ``kappa`` puts
        more mass on extremely small p-values (sharper when the verifier is
        strong, noisier when it is weak); ``kappa = 0.5`` is the usual default.
    higher_is_more_extreme:
        Whether large scores indicate correctness.
    """

    name: str
    null_calibration: np.ndarray
    kappa: float = 0.5
    higher_is_more_extreme: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "null_calibration", np.asarray(self.null_calibration, dtype=np.float64).ravel()
        )
        if self.null_calibration.size == 0:
            raise ValueError("null_calibration must be non-empty")
        if not (0.0 < self.kappa < 1.0):
            raise ValueError("kappa must lie in (0, 1)")

    # -- core API -----------------------------------------------------------

    def pvalue(self, score: float) -> float:
        """Super-uniform conformal p-value under the null."""
        return conformal_pvalue(
            score, self.null_calibration, higher_is_more_extreme=self.higher_is_more_extreme
        )

    def evalue(self, score: float) -> float:
        """Valid e-value: ``E[e] <= 1`` whenever the score is exchangeable
        with the null calibration set."""
        return vovk_wang_calibrator(self.pvalue(score), self.kappa)

    def evalue_from_trajectories(
        self, scores: Sequence[float], rule: str = "mean"
    ) -> float:
        """Aggregate several trajectory scores, then calibrate once.

        Aggregating *before* calibration is deliberate: the conformal rank must
        compare like with like, so the aggregated statistic has to be scored
        against aggregated null statistics (which is what
        ``null_calibration`` is expected to hold).
        """
        return self.evalue(aggregate_scores(scores, rule=rule))

    # -- diagnostics --------------------------------------------------------

    @property
    def n_calibration(self) -> int:
        return int(self.null_calibration.size)

    def describe(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kappa": self.kappa,
            "n_calibration": self.n_calibration,
            "higher_is_more_extreme": self.higher_is_more_extreme,
            "null_mean": float(np.mean(self.null_calibration)),
            "null_std": float(np.std(self.null_calibration)),
        }


@dataclass
class VerifierBank:
    """A set of calibrated verifiers fused into one validity-preserving e-value.

    The fusion rule is the **arithmetic mean** (possibly weighted), which is the
    only combination that stays valid unconditionally.  ``combine_product`` is
    exposed for the (rare) case where conditional independence can be argued.
    """

    verifiers: list[ConformalEVerifier] = field(default_factory=list)
    weights: np.ndarray | None = None
    rule: str = "mean"

    def add(self, v: ConformalEVerifier, weight: float | None = None) -> "VerifierBank":
        self.verifiers.append(v)
        if weight is not None:
            old = (
                np.ones(len(self.verifiers) - 1)
                if self.weights is None
                else np.asarray(self.weights, dtype=np.float64)
            )
            self.weights = np.append(old, float(weight))
        return self

    def evalues(self, scores: dict[str, Sequence[float]] | Sequence[float]) -> np.ndarray:
        """Per-verifier e-values for one candidate.

        ``scores`` maps verifier name -> trajectory scores; a bare sequence is
        broadcast to every verifier (single-verifier convenience path).
        """
        if not self.verifiers:
            return np.array([], dtype=np.float64)
        out = []
        for v in self.verifiers:
            if isinstance(scores, dict):
                s = scores.get(v.name, [])
            else:
                s = scores
            if len(s) == 0:
                out.append(1.0)  # no evidence -> neutral e-value (E=1)
            else:
                out.append(v.evalue_from_trajectories(s))
        return np.asarray(out, dtype=np.float64)

    def fuse(self, scores: dict[str, Sequence[float]] | Sequence[float]) -> float:
        """Fused e-value for one candidate (validity-preserving)."""
        e = self.evalues(scores)
        if e.size == 0:
            return 1.0
        if self.rule == "mean":
            return combine_mean(e, self.weights)
        if self.rule == "product":
            return float(np.prod(e))
        raise ValueError(f"unknown fusion rule: {self.rule!r}")

    def describe(self) -> dict[str, object]:
        return {
            "n_verifiers": len(self.verifiers),
            "rule": self.rule,
            "weights": None if self.weights is None else self.weights.tolist(),
            "verifiers": [v.describe() for v in self.verifiers],
        }


def vce_eprocess(
    verifier: ConformalEVerifier,
    scores: Sequence[float],
    rule: str = "mean",
) -> np.ndarray:
    """Running VCE e-process over a stream of trajectory scores.

    ``E_t = prod_{i<=t} e_i`` would require independence; the **safe** anytime
    object is the running *average* ``E_t = mean_{i<=t} e_i``, which is a valid
    e-value at every ``t`` simultaneously (a mean of e-variables is an
    e-variable; Ville's inequality then applies to any stopping time).
    """
    e = np.asarray([verifier.evalue(float(s)) for s in scores], dtype=np.float64)
    if e.size == 0:
        return np.array([], dtype=np.float64)
    return np.cumsum(e) / np.arange(1, e.size + 1)


def make_verifier(
    name: str,
    null_scores: Sequence[float],
    kappa: float = 0.5,
    rng: np.random.Generator | None = None,
) -> ConformalEVerifier:
    """Convenience constructor (kept explicit about what the null set is)."""
    del rng  # calibration sets are observed data, never synthesised
    return ConformalEVerifier(
        name=name,
        null_calibration=np.asarray(null_scores, dtype=np.float64),
        kappa=kappa,
    )


def calibrator_factory(kappa: float) -> Callable[[float], float]:
    """Expose a bare Vovk-Wang calibrator for tests and notebooks."""
    return lambda p: vovk_wang_calibrator(p, kappa)
