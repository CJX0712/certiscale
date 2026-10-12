"""CertiScale (衡证) -- the flagship: CITE x VCE x AVBA.

What is actually new here
-------------------------
CITE (arXiv:2605.05873) is the strongest known procedure for anytime-valid
certification of the **modal** response.  It is, however, blind in three ways:

1. It never inspects *correctness* -- a model that is consistently wrong is
   certified just as fast as one that is consistently right.
2. It treats each question in isolation -- no notion of a shared compute budget.
3. It emits a single label -- no prediction set with a coverage guarantee.

CertiScale composes CITE with two new components (``verifier.py``,
``allocator.py``) and keeps the guarantee **exact**, not approximate.

The validity argument (why the composition is not a hack)
---------------------------------------------------------
The two components test *different* null hypotheses:

* CITE  : ``H0_mode  : p_r <= sup_{a != r} p_a``  (r is not the unique mode)
* VCE   : ``H0_correct : r is not the correct answer``

CertiScale certifies the conjunction "r is the unique mode **and** r is
correct", so the joint null is the union ``H0 = H0_mode u H0_correct``.  The
stopping rule requires **both** channels to fire:

    tau = inf{ t : E_t >= 3/eps_cite AND L_t > U_t     (CITE channel)
                   AND  W_t >= 1/eps_vce  }            (VCE channel)

In every null world at least one channel is a true null, so

    P(tau < infinity) <= max(eps_cite, eps_vce),

i.e. the error is bounded by the *larger* of the two budgets rather than their
sum -- the conjunction is what buys us that, and it is why we set both to eps.

Why the VCE channel is a genuine e-process (not a running average)
------------------------------------------------------------------
A running *mean* of e-values is an e-variable at each fixed ``t`` but is **not**
a supermartingale, so Ville's inequality would not apply and peeking would
inflate the error.  We therefore use the **product** ``W_t = prod_i e_i``.
Because the sampled trajectories are i.i.d. and the conformal calibration set is
frozen before scoring, each ``e_i`` is a function of ``X_i`` alone, hence

    E[ e_i | F_{i-1} ] = E[ e_i ] <= 1   (conditional e-variable),

which makes ``W_t`` a non-negative supermartingale and Ville's inequality
applies at *any* data-dependent stopping time.  All arithmetic is done in
log-space so the product cannot overflow.

Author: 晨星 (CJX0712)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .allocator import BudgetAllocator
from .cite import CITECertifier, CiteConfig
from .eprocess import safe_log
from .verifier import ConformalEVerifier, VerifierBank

__all__ = ["CertiScaleConfig", "CertiScaleEngine", "CertiScaleSystem"]


def _normalise_scores(scores: dict[str, float] | float | None) -> dict[str, list[float]]:
    """Coerce a per-step score payload into ``{verifier: [values]}`` form."""
    if scores is None:
        return {}
    if isinstance(scores, dict):
        out: dict[str, list[float]] = {}
        for k, v in scores.items():
            if isinstance(v, (list, tuple, np.ndarray)):
                out[k] = [float(x) for x in v]
            else:
                out[k] = [float(v)]
        return out
    return {"_default": [float(scores)]}


@dataclass(frozen=True)
class CertiScaleConfig:
    """Configuration of the flagship.

    ``eps_cite`` and ``eps_vce`` both default to ``eps``; see the module
    docstring for why the joint error is their **max**, not their sum.
    """

    eps: float = 0.05
    eps_cite: float | None = None
    eps_vce: float | None = None
    cite: CiteConfig = field(default_factory=CiteConfig)
    use_verifier: bool = True
    # Trajectories bundled into one conformal statistic before calibration.
    # Batching is the *only* tuning knob that matters for VCE power: it trades
    # responsiveness for variance.  batch = 8 is a good default; batch = 1 is
    # maximally responsive but noisy, and the calibration set MUST be built
    # with the same value.
    vce_batch: int = 8

    def resolved(self) -> tuple[float, float, float]:
        eps_c = self.eps if self.eps_cite is None else self.eps_cite
        eps_v = self.eps if self.eps_vce is None else self.eps_vce
        return self.eps, eps_c, eps_v

    def with_vce_batch(self, batch: int) -> "CertiScaleConfig":
        """Return a copy with a different VCE batch size."""
        return CertiScaleConfig(
            eps=self.eps,
            eps_cite=self.eps_cite,
            eps_vce=self.eps_vce,
            cite=self.cite,
            use_verifier=self.use_verifier,
            vce_batch=int(batch),
        )

    def with_eps(self, eps_cite: float, eps_vce: float) -> "CertiScaleConfig":
        """Return a copy carrying explicit channel budgets."""
        return CertiScaleConfig(
            eps=self.eps,
            eps_cite=eps_cite,
            eps_vce=eps_vce,
            cite=CiteConfig(
                eps=eps_cite,
                lam_pw_lo=self.cite.lam_pw_lo,
                lam_pw_hi=self.cite.lam_pw_hi,
                lam_pw_n=self.cite.lam_pw_n,
                lam_r_lo=self.cite.lam_r_lo,
                lam_r_hi=self.cite.lam_r_hi,
                lam_r_n=self.cite.lam_r_n,
                lcb_probes=self.cite.lcb_probes,
            ),
            use_verifier=self.use_verifier,
        )


class CertiScaleEngine:
    """Single-question certification engine (CITE channel x VCE channel).

    One ``step`` consumes **one sampled trajectory**: its final answer label
    and (optionally) the verifier scores attached to that trajectory.
    """

    def __init__(
        self,
        config: CertiScaleConfig,
        target: str,
        verifiers: list[ConformalEVerifier] | VerifierBank | None = None,
    ):
        _, eps_c, eps_v = config.resolved()
        self.cfg = config
        self.target = target
        self.eps_cite = eps_c
        self.eps_vce = eps_v
        self.cite_cfg = config.cite if config.cite.eps == eps_c else CiteConfig(
            eps=eps_c,
            lam_pw_lo=config.cite.lam_pw_lo,
            lam_pw_hi=config.cite.lam_pw_hi,
            lam_pw_n=config.cite.lam_pw_n,
            lam_r_lo=config.cite.lam_r_lo,
            lam_r_hi=config.cite.lam_r_hi,
            lam_r_n=config.cite.lam_r_n,
            lcb_probes=config.cite.lcb_probes,
        )
        self.cite = CITECertifier(self.cite_cfg, target=target)
        self.bank = (
            verifiers
            if isinstance(verifiers, VerifierBank)
            else (VerifierBank(verifiers=list(verifiers or [])))
        )
        # log-wealth of the VCE channel (log-space product of e-values)
        self.log_wealth = 0.0
        self.n_vce = 0
        self.vce_batch = max(int(config.vce_batch), 1)
        # trajectories waiting to be bundled into the next conformal statistic
        self._buffer: list[dict[str, list[float]]] = []
        self.certified = False
        self.tau: int | None = None
        self.channel_cite = False
        self.channel_vce = False

    # -- main loop ----------------------------------------------------------

    def step(
        self,
        label: str,
        verifier_scores: dict[str, float] | float | None = None,
    ) -> bool:
        """Ingest one sampled trajectory; return ``True`` when certified."""
        cite_fired = self.cite.step(label)
        self.channel_cite = cite_fired

        vce_fired = self._update_vce(label, verifier_scores)

        if cite_fired and vce_fired:
            self.certified = True
            self.tau = self.cite.state.t
            return True
        return False

    def _update_vce(
        self,
        label: str,
        verifier_scores: dict[str, float] | float | None,
    ) -> bool:
        """Advance the VCE channel.

        Only trajectories that actually *support the target* carry evidence
        about "r is correct"; a trajectory ending in some other answer is
        uninformative for that hypothesis and is skipped (this is what keeps
        the conditional-e-variable property intact).
        """
        if not self.cfg.use_verifier or not self.bank.verifiers:
            # No verifier -> channel is vacuously satisfied (pure CITE mode).
            self.channel_vce = True
            return True
        if label != self.target or verifier_scores is None:
            return bool(self.channel_vce and self.log_wealth >= self._vce_threshold_log())

        self._buffer.append(_normalise_scores(verifier_scores))
        if len(self._buffer) < self.vce_batch:
            # Batch not full yet: no new evidence this step.  The channel keeps
            # its previous state, which is exactly what "no evidence" means for
            # a supermartingale (the multiplier is 1).
            return bool(self.channel_vce and self.log_wealth >= self._vce_threshold_log())

        # Bundle the batch into ONE statistic, then calibrate once.  Averaging
        # inside the batch shrinks the variance of the conformal statistic by
        # ~1/batch, which is what turns a weak verifier into a usable one; the
        # batches themselves are disjoint, so the product over batches is still
        # a supermartingale and Ville's inequality still applies.
        agg: dict[str, list[float]] = {}
        for item in self._buffer:
            for k, v in item.items():
                agg.setdefault(k, []).extend(v)
        self._buffer = []

        e = float(self.bank.fuse(agg))
        if e <= 0.0:
            e = 1e-12
        self.log_wealth += safe_log(e)
        self.n_vce += 1
        thr = self._vce_threshold_log()
        self.channel_vce = self.log_wealth >= thr
        return self.channel_vce

    def _vce_threshold_log(self) -> float:
        return -safe_log(self.eps_vce)

    # -- diagnostics --------------------------------------------------------

    @property
    def wealth(self) -> float:
        """Current VCE e-value ``W_t`` (exp of log-wealth, clipped)."""
        return float(np.exp(np.clip(self.log_wealth, -700.0, 700.0)))

    @property
    def log_progress(self) -> float:
        """Signed distance to the certification boundary (<= 0 means keep going).

        Used by the budget allocator as the per-question "wealth" signal.
        """
        cite_gap = safe_log(max(self.cite.state.last_E, 1e-300)) - safe_log(
            self.cite_cfg.pairwise_threshold
        )
        vce_gap = self.log_wealth - self._vce_threshold_log() if self.cfg.use_verifier and self.bank.verifiers else 0.0
        return float(min(cite_gap, vce_gap))

    def audit(self) -> dict[str, object]:
        a = self.cite.audit()
        a.update(
            {
                "log_wealth_vce": self.log_wealth,
                "wealth_vce": self.wealth,
                "n_vce": self.n_vce,
                "channel_cite": self.channel_cite,
                "channel_vce": self.channel_vce,
                "certified": self.certified,
                "tau": self.tau,
                "eps_cite": self.eps_cite,
                "eps_vce": self.eps_vce,
            }
        )
        return a


class CertiScaleSystem:
    """Multi-question system: K engines sharing one sampling budget via AVBA.

    This is the object a deployment actually drives: hand it K questions, a
    total sample budget, and it decides both *who* to sample next and *when*
    each question is settled.
    """

    def __init__(
        self,
        config: CertiScaleConfig,
        targets: list[str],
        verifiers: list[ConformalEVerifier] | None = None,
        budget: int = 1000,
        batch_size: int = 1,
        strategy: str = "wga",
    ):
        self.cfg = config
        self.targets = list(targets)
        self.engines = [
            CertiScaleEngine(config, target=t, verifiers=verifiers) for t in self.targets
        ]
        self.allocator = BudgetAllocator(
            n_problems=len(self.targets),
            budget=budget,
            batch_size=batch_size,
            strategy=strategy,
        )

    def wealth_vector(self) -> np.ndarray:
        return np.array([e.log_progress for e in self.engines], dtype=np.float64)

    def run(
        self,
        sampler: "callable",
        max_rounds: int | None = None,
    ) -> dict[str, object]:
        """Drive the whole system.

        Parameters
        ----------
        sampler:
            ``sampler(i) -> (label, verifier_scores)``: draw one trajectory for
            question ``i``.  Must be reproducible for benchmark use.
        max_rounds:
            Safety cap on the number of allocation rounds.
        """
        alloc = self.allocator
        rounds = 0
        while not alloc.exhausted:
            if max_rounds is not None and rounds >= max_rounds:
                break
            alloc.observe(self.wealth_vector())
            idx = alloc.allocate(self.wealth_vector())
            if idx.size == 0:
                break
            for i in idx:
                label, scores = sampler(int(i))
                eng = self.engines[int(i)]
                eng.step(label, scores)
                alloc.state.n_draws[int(i)] += 1
                alloc.consume(1)
                if eng.certified:
                    alloc.mark_done(int(i))
            rounds += 1
        return self.summary()

    def summary(self) -> dict[str, object]:
        n_cert = sum(1 for e in self.engines if e.certified)
        taus = [e.tau for e in self.engines if e.tau is not None]
        return {
            "strategy": self.allocator.strategy,
            "n_problems": len(self.engines),
            "budget": self.allocator.budget,
            "spent": self.allocator.spent,
            "n_certified": int(n_cert),
            "certification_rate": float(n_cert / max(len(self.engines), 1)),
            "mean_tau": float(np.mean(taus)) if taus else float("nan"),
            "median_tau": float(np.median(taus)) if taus else float("nan"),
            "draws_per_problem": self.allocator.state.n_draws.tolist(),
        }
