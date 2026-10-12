"""CITE -- Certification by Intersection-union Testing with E-processes.

Faithful re-implementation of the current world-class SOTA for anytime-valid
certification of the modal response of an LLM under self-consistency sampling:

    Ota, Iwase, Ichihara, Komiyama & Imaizumi (2026),
    "CITE: Anytime-Valid Statistical Inference in LLM Self-Consistency",
    arXiv:2605.05873.

Why this file exists (zero-rewrite principle)
---------------------------------------------
CITE is, to the best of current knowledge, the strongest published procedure
for *"stop sampling whenever you like and still control the false-certification
rate"* on an unknown, possibly countably infinite answer alphabet.  The goal of
this repository is **not** to reinvent it -- it is to reuse it verbatim as the
Tier-0 baseline and then strictly improve on it (see ``flagship.py``).

The three components reproduced here are CITE Def. 3.1 (pairwise mixture
e-process), Def. 3.3 (lower confidence bound on ``p_r``) and Def. 3.5 (unseen
upper bound), combined by the stopping rule of Def. 3.7:

    tau = inf{ t >= 1 : E_t >= 3/eps  AND  L_t > U_t }

Author: 晨星 (CJX0712)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .eprocess import (
    geometric_grid,
    lcb_lower_bound,
    lcb_process,
    mixture_pairwise_eprocess,
    unseen_bound,
    uniform_grid_weights,
)

__all__ = ["CiteConfig", "CiteState", "CITECertifier"]


@dataclass(frozen=True)
class CiteConfig:
    """Hyper-parameters of CITE.

    Defaults follow the grid requirements of CITE Condition 4.2, which asks the
    pairwise grid to contain a ``lambda`` in ``[delta/8, delta/4]`` and the LCB
    grid a ``lambda`` in ``[1/32, 1/16]``.  Since ``delta`` is unknown a priori,
    we cover the full plausible range with a geometric grid.
    """

    eps: float = 0.05
    # pairwise betting grid: (0, 1), geometric, dense near 0
    lam_pw_lo: float = 1e-3
    lam_pw_hi: float = 0.9
    lam_pw_n: int = 24
    # LCB grid: (0, inf)
    lam_r_lo: float = 1.0 / 64.0
    lam_r_hi: float = 8.0
    lam_r_n: int = 24
    # number of bisection probes used by the LCB solver
    lcb_probes: int = 60

    def grids(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        g_pw = geometric_grid(self.lam_pw_lo, self.lam_pw_hi, self.lam_pw_n)
        g_r = geometric_grid(self.lam_r_lo, self.lam_r_hi, self.lam_r_n)
        return g_pw, uniform_grid_weights(g_pw), g_r, uniform_grid_weights(g_r)

    @property
    def pairwise_threshold(self) -> float:
        """CITE uses ``3/eps`` because the error is split three ways."""
        return 3.0 / self.eps


@dataclass
class CiteState:
    """Mutable per-question state of a running CITE procedure."""

    target: str
    t: int = 0
    counts: dict[str, int] = field(default_factory=dict)
    n_target: int = 0
    # diagnostics of the most recent step
    last_E: float = 1.0
    last_L: float = 0.0
    last_U: float = 1.0
    certified: bool = False
    tau: int | None = None

    @property
    def runner_up(self) -> tuple[str | None, int]:
        """Empirical runner-up ``\\hat a_t = argmax_{a != r} N_t(a)``."""
        best_a, best_n = None, -1
        for a, n in self.counts.items():
            if a == self.target:
                continue
            if n > best_n:
                best_a, best_n = a, n
        return best_a, max(best_n, 0)

    @property
    def phat(self) -> float:
        return self.n_target / self.t if self.t else 0.0


class CITECertifier:
    """Anytime-valid certifier for the fixed-target unique-mode hypothesis.

    Usage
    -----
    >>> cert = CITECertifier(CiteConfig(eps=0.05), target="42")
    >>> for x in samples:                 # x is one sampled answer label
    ...     if cert.step(x):
    ...         break                     # certified "42" as the unique mode
    """

    def __init__(self, config: CiteConfig, target: str):
        if not (0.0 < config.eps < 1.0):
            raise ValueError("eps must lie in (0, 1)")
        self.cfg = config
        self.state = CiteState(target=target)
        self._g_pw, self._w_pw, self._g_r, self._w_r = config.grids()
        self._thr = config.pairwise_threshold

    # -- single-step update -------------------------------------------------

    def step(self, x: str) -> bool:
        """Ingest one sampled label; return ``True`` when certified.

        The three components are re-evaluated from scratch at every ``t``; all
        of them admit closed forms in ``(N_t(r), N_t(a), t)``, so the per-step
        cost is ``O(|A_t| + |Lambda|)`` rather than ``O(t)``.
        """
        st = self.state
        st.t += 1
        st.counts[x] = st.counts.get(x, 0) + 1
        if x == st.target:
            st.n_target += 1

        # (iii) unseen upper bound -- depends on t and eps only
        st.last_U = unseen_bound(st.t, self.cfg.eps)

        # (ii) lower confidence bound on p_r
        st.last_L = lcb_lower_bound(
            st.n_target,
            st.t,
            self._g_r,
            self._w_r,
            threshold=self._thr,
            n_probe=self.cfg.lcb_probes,
        )

        # (i) pairwise e-process against the empirical runner-up
        a_hat, n_hat = st.runner_up
        if a_hat is None:
            # No observed competitor: pairwise condition is vacuously true and
            # the unseen bound alone governs whether certification happens.
            st.last_E = float("inf")
        else:
            st.last_E = mixture_pairwise_eprocess(
                st.n_target, n_hat, self._g_pw, self._w_pw
            )

        if st.last_E >= self._thr and st.last_L > st.last_U:
            st.certified = True
            st.tau = st.t
            return True
        return False

    # -- batch helpers ------------------------------------------------------

    def run(self, samples: Sequence[str], max_t: int | None = None) -> CiteState:
        """Feed a whole sample path; stop early once certified."""
        limit = len(samples) if max_t is None else min(len(samples), max_t)
        for i in range(limit):
            if self.step(samples[i]):
                break
        return self.state

    # -- diagnostics --------------------------------------------------------

    def audit(self) -> dict[str, float | int | bool | str | None]:
        st = self.state
        a_hat, n_hat = st.runner_up
        return {
            "t": st.t,
            "N_r": st.n_target,
            "phat_r": st.phat,
            "runner_up": a_hat,
            "N_runner_up": n_hat,
            "E_t": st.last_E,
            "threshold": self._thr,
            "L_t": st.last_L,
            "U_t": st.last_U,
            "certified": st.certified,
            "tau": st.tau,
        }

    # -- exposed for tests / theory checks ----------------------------------

    def lcb_at(self, q: float, t: int | None = None, n_target: int | None = None) -> float:
        st = self.state
        tt = st.t if t is None else t
        nn = st.n_target if n_target is None else n_target
        return lcb_process(nn, tt, q, self._g_r, self._w_r)
