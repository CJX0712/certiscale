"""e-value / e-process primitives (Layer L2: mathematical foundations).

Reuse-first provenance (zero-rewrite principle)
----------------------------------------------
Every object here is a *known-optimal* construction from the world-class
literature, re-implemented in closed form on top of NumPy:

* Testing-by-betting / pairwise e-process
  -> Ramdas, Ruf, Larsson & Koolen (2023) "Testing exchangeability:
     fork-convexity, supermartingales and e-processes" (arXiv:2209.11119).
  -> Ota, Iwase, Ichihara, Komiyama & Imaizumi (2026) "CITE: Anytime-Valid
     Statistical Inference in LLM Self-Consistency" (arXiv:2605.05873).
* p-to-e calibration
  -> Vovk & Wang (2021) "E-values: calibration, combination and applications"
     (Annals of Statistics 49(3)).
* Conformal p-values
  -> Vovk, Gammerman & Shafer (2005) "Algorithmic Learning in a Random World".

Nothing here is invented from scratch; the *novel* part of this repository is
the composition (see ``flagship.py``), not the primitives.

Author: 晨星 (CJX0712)
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, Sequence

import numpy as np

__all__ = [
    "pairwise_eprocess",
    "lcb_process",
    "unseen_bound",
    "vovk_wang_calibrator",
    "combine_mean",
    "combine_product",
    "geometric_grid",
    "uniform_grid_weights",
    "safe_log",
]


# ---------------------------------------------------------------------------
# numeric helpers
# ---------------------------------------------------------------------------


def safe_log(x: float) -> float:
    """``log`` with a floor at the smallest positive normal double."""
    return math.log(max(float(x), 1e-300))


_OV_LOG = 700.0  # exp(700) ~ 1e304, just inside double range


def _logsumexp(a: np.ndarray) -> float:
    """Numerically stable ``log sum exp`` (the canonical stable primitive)."""
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0:
        return -math.inf
    m = float(np.max(a))
    if not np.isfinite(m):
        return m
    return m + float(np.log(np.sum(np.exp(a - m))))


def _exp_clip(logx: float) -> float:
    """``exp`` that saturates to ``+inf`` instead of overflowing.

    An e-value legitimately grows without bound (that is what makes it
    composable), so saturating above ``exp(700)`` is the correct behaviour:
    every downstream use is a comparison against a finite threshold.
    """
    if logx > _OV_LOG:
        return math.inf
    return float(math.exp(logx))


def geometric_grid(lo: float, hi: float, n: int) -> np.ndarray:
    """Geometric grid on ``[lo, hi]`` with ``n`` points (ascending)."""
    if n <= 0:
        raise ValueError("n must be positive")
    if lo <= 0.0:
        raise ValueError("geometric grid requires lo > 0")
    return np.geomspace(lo, hi, num=int(n), dtype=np.float64)


def uniform_grid_weights(grid: Sequence[float]) -> np.ndarray:
    """Uniform prior over a grid. Weights sum to 1 (<= 1, per CITE Def. 3.1)."""
    g = np.asarray(grid, dtype=np.float64)
    if g.size == 0:
        raise ValueError("empty grid")
    return np.full(g.size, 1.0 / g.size, dtype=np.float64)


# ---------------------------------------------------------------------------
# Component (i): pairwise e-process  --  CITE Eq. (3)-(4)
# ---------------------------------------------------------------------------


def pairwise_eprocess(
    n_target: int | np.ndarray,
    n_competitor: int | np.ndarray,
    lam: float | np.ndarray,
) -> np.ndarray:
    """Pairwise (target vs. competitor) e-process, CITE Eq. (3).

    ``E_t^(r,a)(lambda) = prod_i (1 + lambda * Z_i) = (1+lambda)^N_r (1-lambda)^N_a``
    with ``Z_i = 1{X_i = r} - 1{X_i = a} in {-1, 0, +1}``.

    Under ``H0: p_r <= p_a`` this is a non-negative supermartingale (NSM) with
    initial value 1, hence Ville's inequality gives
    ``P(sup_t E_t >= 1/alpha) <= alpha``.

    Parameters
    ----------
    n_target, n_competitor:
        Counts ``N_t(r)`` and ``N_t(a)``. Broadcast against ``lam``.
    lam:
        Betting parameter in ``(0, 1)``.

    Returns
    -------
    Array of e-values (same broadcast shape).
    """
    lt = np.asarray(lam, dtype=np.float64)
    if np.any(lt <= 0.0) or np.any(lt >= 1.0):
        raise ValueError("lambda must lie in (0, 1) so that every factor stays positive")
    nr = np.asarray(n_target, dtype=np.float64)
    na = np.asarray(n_competitor, dtype=np.float64)
    return _exp_clip(np.log1p(lt) * nr + np.log1p(-lt) * na)


def log_pairwise_eprocess(
    n_target: int | np.ndarray,
    n_competitor: int | np.ndarray,
    lam: float | np.ndarray,
) -> np.ndarray:
    """Log-space pairwise e-process (stable for large ``t``).

    ``log E = N_r log(1+lambda) + N_a log(1-lambda)``; ``log1p`` is used on
    both branches to avoid cancellation when ``lambda`` approaches 0 or 1.
    """
    lt = np.asarray(lam, dtype=np.float64)
    if np.any(lt <= 0.0) or np.any(lt >= 1.0):
        raise ValueError("lambda must lie in (0, 1) so that every factor stays positive")
    nr = np.asarray(n_target, dtype=np.float64)
    na = np.asarray(n_competitor, dtype=np.float64)
    return np.log1p(lt) * nr + np.log1p(-lt) * na


def mixture_pairwise_eprocess(
    n_target: int,
    n_competitor: int,
    grid: Sequence[float],
    weights: Sequence[float] | None = None,
) -> float:
    """Mixture (lambda-integrated) pairwise e-process, CITE Def. 3.1 / Eq. (4).

    A convex combination of NSMs is an NSM, so mixing over a *data-independent*
    grid preserves anytime validity even though the argmax runner-up
    ``\\hat a_t`` is data dependent (the *fixed*-competitor process is what is
    bounded; see CITE Cor. 3.2 and the proof of Thm. 4.1).

    Computed in log-space via ``logsumexp`` because a single grid point can
    carry a factor ``(1+lambda)^N`` far beyond double range even when the
    mixture itself is modest.
    """
    g = np.asarray(grid, dtype=np.float64)
    w = uniform_grid_weights(g) if weights is None else np.asarray(weights, dtype=np.float64)
    lg = log_pairwise_eprocess(n_target, n_competitor, g) + np.log(np.asarray(w))
    return _exp_clip(_logsumexp(lg))


# ---------------------------------------------------------------------------
# Component (ii): lower confidence bound on p_r  --  CITE Eq. (5), Def. 3.3
# ---------------------------------------------------------------------------


def lcb_process(
    n_target: int,
    t: int,
    q: float,
    grid: Sequence[float],
    weights: Sequence[float] | None = None,
) -> float:
    """Mixture e-process ``M_t(q)`` for the hypothesis ``p_r <= q``.

    Closed form (derived, not approximated).  Each factor is
    ``1 + lambda (1{X_i = r} - q)``, so over ``t`` draws the ``N_t(r)`` hits
    contribute ``(1 + lambda(1-q))`` and the remaining ``t - N_t(r)`` misses
    contribute ``(1 - lambda q)``:

    ``M_t(q) = sum_lambda v_lambda (1 + lambda(1-q))^{N_r} (1 - lambda q)^{t-N_r}``

    Only lambdas with ``lambda < 1/q`` are admissible (CITE Def. 3.3), which
    keeps every factor strictly positive.
    """
    if t < 0:
        raise ValueError("t must be non-negative")
    n_target = int(n_target)
    if n_target > t:
        raise ValueError("N_t(r) cannot exceed t")
    q = float(q)
    if not (0.0 < q <= 1.0):
        raise ValueError("q must lie in (0, 1]")

    g = np.asarray(grid, dtype=np.float64)
    w = uniform_grid_weights(g) if weights is None else np.asarray(weights, dtype=np.float64)
    keep = g < 1.0 / q  # admissibility: 1 - lambda*q > 0
    if not np.any(keep):
        return 0.0
    lam = g[keep]
    ww = w[keep]
    ww = ww / ww.sum()  # renormalise on the admissible sub-grid

    # log-space: log(1 + lambda(1-q)) and log(1 - lambda q) via log1p
    lg = (
        np.log(ww)
        + np.log1p(lam * (1.0 - q)) * float(n_target)
        + np.log1p(-lam * q) * float(t - n_target)
    )
    return _exp_clip(_logsumexp(lg))


def lcb_lower_bound(
    n_target: int,
    t: int,
    grid: Sequence[float],
    weights: Sequence[float] | None = None,
    threshold: float = 30.0,
    n_probe: int = 200,
) -> float:
    """Time-uniform lower confidence bound ``L_t(r)`` (CITE Def. 3.3).

    ``L_t = sup( {0} u {q in (0, phat_t(r)] : M_t(q) >= threshold} )``

    ``q -> M_t(q)`` is non-increasing (CITE Lemma C.3), so the superlevel set
    is a downset and the supremum is well posed.  We bisect on a probe grid
    inside ``(0, phat_t(r)]`` rather than scanning unbounded q.
    """
    if t <= 0:
        return 0.0
    phat = n_target / float(t)
    if phat <= 0.0:
        return 0.0
    lo, hi = 0.0, phat
    # M_t is non-increasing in q: M_t(lo) >= M_t(hi).  If even M_t(phat) clears
    # the threshold the whole interval is admissible -> L_t = phat.
    if lcb_process(n_target, t, hi, grid, weights) >= threshold:
        return float(hi)
    if lcb_process(n_target, t, lo + 1e-12, grid, weights) < threshold:
        return 0.0
    for _ in range(int(n_probe)):
        mid = 0.5 * (lo + hi)
        if lcb_process(n_target, t, mid, grid, weights) >= threshold:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12:
            break
    return float(lo)


# ---------------------------------------------------------------------------
# Component (iii): unseen-category upper bound  --  CITE Def. 3.5
# ---------------------------------------------------------------------------


def unseen_bound(t: int, eps: float = 0.05) -> float:
    """``U_t = min{ u in (0,1] : u^{-1} (1-u)^t <= eps/3 }``.

    Probability-weighted union bound: a category of mass ``u`` survives ``t``
    i.i.d. draws with probability ``(1-u)^t``; the ``u^{-1}`` factor pays the
    union-bound cost over the (unknown, possibly infinite) rare categories.
    Monotone decreasing in ``t``; solved by bisection on ``(0, 1]``.
    """
    if t <= 0:
        return 1.0
    budget = eps / 3.0
    lo, hi = 1e-12, 1.0

    def viol(u: float) -> float:
        # u -> 1 makes (1-u)^t -> 0, so the constraint is trivially satisfied.
        if u >= 1.0:
            return -math.inf
        return math.log(1.0 - u) * t - safe_log(u) - safe_log(budget)

    # ``viol`` is decreasing in u, so the feasible set is an up-set:
    #   * viol(lo) <= 0  -> lo itself is the minimum feasible u;
    #   * viol(hi) >  0  -> infeasible on (0, 1], fall back to the trivial bound 1.
    if viol(lo) <= 0.0:
        return float(lo)
    if viol(hi) > 0.0:
        return float(hi)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if viol(mid) <= 0.0:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-14:
            break
    return float(hi)


# ---------------------------------------------------------------------------
# p-to-e calibration  --  Vovk & Wang (2021)
# ---------------------------------------------------------------------------


def vovk_wang_calibrator(p: float, kappa: float = 0.5) -> float:
    """Convert a (super-uniform) p-value into a valid e-value.

    ``e_kappa(p) = kappa * p^(kappa - 1)`` for ``kappa in (0, 1)``.

    Validity: ``e_kappa`` is non-negative, non-increasing on ``(0, 1]``, and
    ``int_0^1 e_kappa(u) du = 1``.  For any ``p`` that is super-uniform
    (``P(p <= u) <= u``, which is exactly what a conformal p-value gives under
    exchangeability) ``p`` stochastically dominates ``U[0,1]``, and a
    non-increasing score preserves that order, hence ``E[e_kappa(p)] <= 1``.
    That is precisely the definition of an e-variable.

    Note ``e -> +inf`` as ``p -> 0``: overwhelming evidence is unbounded, which
    is what makes e-values composable by averaging.
    """
    p = float(p)
    if not (0.0 < p <= 1.0):
        # p == 0 (impossible score) is *infinite* evidence; clip to a large
        # finite value so downstream arithmetic stays finite.
        if p <= 0.0:
            return 1e12
        raise ValueError("p must lie in [0, 1]")
    if not (0.0 < kappa < 1.0):
        raise ValueError("kappa must lie in (0, 1)")
    return float(kappa * math.pow(p, kappa - 1.0))


# ---------------------------------------------------------------------------
# e-value combination rules (validity-preserving)
# ---------------------------------------------------------------------------


def combine_mean(e_values: Iterable[float], weights: Sequence[float] | None = None) -> float:
    """Arithmetic (weighted) mean of e-values.

    Always validity-preserving: a convex combination of e-variables is an
    e-variable, with **no independence assumption**.  This is the conservative
    default used by CertiScale when verifiers may share information.
    """
    e = np.asarray(list(e_values), dtype=np.float64)
    if e.size == 0:
        return 1.0
    if np.any(e < 0.0):
        raise ValueError("e-values must be non-negative")
    if weights is None:
        return float(e.mean())
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != e.shape:
        raise ValueError("weights must match e_values")
    s = w.sum()
    if s <= 0.0:
        raise ValueError("weights must have positive sum")
    return float(np.dot(w / s, e))


def combine_product(e_values: Iterable[float]) -> float:
    """Product of *independent* e-values (a martingale under the null).

    Valid only under conditional independence of the component e-values; use
    :func:`combine_mean` when that cannot be certified.  CertiScale records
    which rule it used so the guarantee stays auditable.
    """
    e = np.asarray(list(e_values), dtype=np.float64)
    if e.size == 0:
        return 1.0
    if np.any(e < 0.0):
        raise ValueError("e-values must be non-negative")
    return float(np.exp(np.sum(np.log(np.clip(e, 1e-300, None)))))


def make_calibrator(kappa: float) -> Callable[[float], float]:
    """Return a fixed-``kappa`` Vovk-Wang calibrator closure."""
    def _f(p: float) -> float:
        return vovk_wang_calibrator(p, kappa)
    return _f
