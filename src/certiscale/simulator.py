"""Reproducible synthetic question generator for benchmarks and invariant tests.

Everything here is *simulation only* -- no certification logic depends on it,
and no simulator quantity is ever visible to a certifier (that would be
peeking).  The distributions are chosen to mirror what real LLM
self-consistency sampling looks like:

* answer distributions are heavy-tailed (Zipf), matching the diffuse-tail
  regimes studied in CITE Sec. 6 (Zipf 1.1-1.3 over hundreds of labels);
* the modal answer is **not** always the correct one (that is precisely the
  failure mode VCE exists to catch);
* verifier scores are noisy and only partially informative.

Author: 晨星 (CJX0712)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

__all__ = ["SimConfig", "SyntheticProblem", "make_problem_bank", "zipf_probs", "verifier_calibration"]


def zipf_probs(n_labels: int, exponent: float = 1.1, rng: np.random.Generator | None = None) -> np.ndarray:
    """Zipf probabilities over ``n_labels`` categories, normalised.

    ``exponent`` close to 1 gives the very diffuse tails that make mode
    certification genuinely hard (CITE's "LLM-like" regime).
    """
    if n_labels <= 0:
        raise ValueError("n_labels must be positive")
    ranks = np.arange(1, n_labels + 1, dtype=np.float64)
    w = np.power(ranks, -exponent)
    if rng is not None:
        # multiplicative jitter keeps the tail diffuse but non-degenerate
        w = w * rng.uniform(0.7, 1.3, size=n_labels)
    return w / w.sum()


@dataclass(frozen=True)
class SimConfig:
    """Parameters of one synthetic benchmark regime."""

    n_labels: int = 200
    zipf_exponent: float = 1.2
    # probability mass of the top answer
    top_mass: float | None = 0.35
    # probability that the modal answer is actually the correct one
    truth_is_mode: bool = True
    # verifier quality: separation (in std units) between correct and wrong
    verifier_sep: float = 1.0
    verifier_noise: float = 1.0
    n_verifiers: int = 2
    seed: int = 0


@dataclass
class SyntheticProblem:
    """One question: a categorical answer distribution + a noisy verifier."""

    labels: list[str]
    probs: np.ndarray
    truth: str
    mode: str
    verifier_sep: float = 1.0
    verifier_noise: float = 1.0
    n_verifiers: int = 2
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    # -- sampling -----------------------------------------------------------

    def sample_label(self) -> str:
        i = int(self.rng.choice(len(self.labels), p=self.probs))
        return self.labels[i]

    def verifier_scores(self, label: str) -> dict[str, float]:
        """Noisy per-verifier scores for one trajectory ending in ``label``.

        Correct answers score ``+sep`` on average, wrong answers ``0``; each
        verifier is an independent noisy read-out with unit-scale noise.  The
        separation is deliberately modest -- a verifier that were perfect would
        make the whole problem trivial and the benchmark meaningless.
        """
        out: dict[str, float] = {}
        correct = 1.0 if label == self.truth else 0.0
        for k in range(self.n_verifiers):
            mu = self.verifier_sep * correct
            out[f"v{k}"] = float(
                self.rng.normal(loc=mu, scale=self.verifier_noise)
            )
        return out

    def sample(self) -> tuple[str, dict[str, float]]:
        """Draw one full trajectory: an answer label plus its verifier scores."""
        lab = self.sample_label()
        return lab, self.verifier_scores(lab)

    # -- ground truth (benchmark-only, never exposed to a certifier) --------

    @property
    def truth_is_mode(self) -> bool:
        return self.truth == self.mode

    def p_truth(self) -> float:
        i = self.labels.index(self.truth)
        return float(self.probs[i])

    def p_mode(self) -> float:
        i = self.labels.index(self.mode)
        return float(self.probs[i])


def make_problem_bank(
    n_problems: int,
    cfg: SimConfig,
    seed: int | None = None,
) -> list[SyntheticProblem]:
    """Build ``n_problems`` independent synthetic questions."""
    seed = cfg.seed if seed is None else seed
    root = np.random.default_rng(seed)
    out: list[SyntheticProblem] = []
    for j in range(n_problems):
        rng = np.random.default_rng(int(root.integers(0, 2**31 - 1)))
        labels = [f"a{i}" for i in range(cfg.n_labels)]
        probs = zipf_probs(cfg.n_labels, cfg.zipf_exponent, rng=rng)
        if cfg.top_mass is not None:
            probs = _force_top_mass(probs, cfg.top_mass)
        mode = labels[int(np.argmax(probs))]
        if cfg.truth_is_mode:
            truth = mode
        else:
            # pick a non-modal label as the (unknown-to-the-model) truth
            cand = [i for i in range(cfg.n_labels) if labels[i] != mode]
            truth = labels[int(rng.choice(cand))]
        out.append(
            SyntheticProblem(
                labels=labels,
                probs=probs,
                truth=truth,
                mode=mode,
                verifier_sep=cfg.verifier_sep,
                verifier_noise=cfg.verifier_noise,
                n_verifiers=cfg.n_verifiers,
                rng=rng,
            )
        )
    return out


def _force_top_mass(probs: np.ndarray, top_mass: float) -> np.ndarray:
    """Rescale the tail so the top category carries exactly ``top_mass``."""
    p = np.asarray(probs, dtype=np.float64).copy()
    k = int(np.argmax(p))
    top = p[k]
    tail = np.delete(p, k)
    s = tail.sum()
    if s <= 0:
        return p
    new = np.empty_like(p)
    new[k] = top_mass
    idx = [i for i in range(p.size) if i != k]
    new[idx] = tail / s * (1.0 - top_mass)
    return new


def verifier_calibration(
    problems: Sequence[SyntheticProblem],
    n_samples: int,
    seed: int = 0,
    verifier_index: int = 0,
    batch: int = 1,
) -> np.ndarray:
    """Build a **null** calibration set for one verifier.

    The null population is "scores of *wrong* answers".  Sampling it from
    held-out problems (never from the problem being certified) is what makes
    the resulting conformal p-values exchangeable rather than circular.

    Critical consistency requirement
    --------------------------------
    ``batch`` **must** match the batch size the engine aggregates over at test
    time.  A calibration set built from single scores while the engine scores
    batch means compares two different statistics, and the conformal p-value
    silently loses its super-uniformity -- the single most common way this
    construction is botched in practice.
    """
    if batch < 1:
        raise ValueError("batch must be >= 1")
    rng = np.random.default_rng(seed)
    scores: list[float] = []
    if not problems:
        raise ValueError("need at least one problem to calibrate")
    key = f"verifier_index"
    del key
    vk = f"v{verifier_index}"
    for _ in range(int(n_samples)):
        p = problems[int(rng.integers(0, len(problems)))]
        wrong = [l for l in p.labels if l != p.truth]
        lab = wrong[int(rng.integers(0, len(wrong)))]
        vals = [p.verifier_scores(lab)[vk] for _ in range(int(batch))]
        scores.append(float(np.mean(vals)))
    return np.asarray(scores, dtype=np.float64)
