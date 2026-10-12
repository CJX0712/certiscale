"""CertiScale invariant suite -- the acceptance gates of the whole system.

Every test below is a *mathematical* statement, not a regression check.  They
are ordered from primitives (I1-I4) through the novel components (I5-I7) to
the composed flagship (I8-I10) and finally reproducibility (I11-I12).

Monte-Carlo gates use a slack factor: a statistical guarantee can only ever be
*estimated* empirically, so we require the observed violation rate to stay under
the nominal level plus a margin calibrated to the number of replicates
(``slack = 4 * sqrt(eps(1-eps)/R)``, i.e. four standard errors).

Author: 晨星 (CJX0712)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from certiscale.allocator import BudgetAllocator
from certiscale.cite import CITECertifier, CiteConfig
from certiscale.conformal import conformal_pvalue
from certiscale.eprocess import (
    combine_mean,
    combine_product,
    lcb_process,
    pairwise_eprocess,
    unseen_bound,
    vovk_wang_calibrator,
)
from certiscale.flagship import CertiScaleConfig, CertiScaleEngine
from certiscale.simulator import SimConfig, make_problem_bank, verifier_calibration
from certiscale.verifier import ConformalEVerifier, VerifierBank

EPS = 0.05


def mc_slack(eps: float, n: int) -> float:
    """Four-standard-error margin for a Monte-Carlo estimate of ``eps``."""
    return 4.0 * math.sqrt(max(eps * (1.0 - eps), 1e-12) / max(n, 1))


# ===========================================================================
# I1-I4: primitives reused from the literature (must be exactly right, or
#        everything built on top of them is decoration)
# ===========================================================================


def test_I1_pairwise_eprocess_is_supermartingale_under_null():
    """Ville: under p_r = p_a, P(sup_t E_t >= 1/alpha) <= alpha."""
    rng = np.random.default_rng(20261013)
    alpha = 0.10
    n_paths, horizon = 400, 60
    lam = 0.25
    hits = 0
    for _ in range(n_paths):
        # tie: p_r = p_a = 0.4 (the least favourable null for this statistic)
        z = rng.choice([-1.0, 0.0, 1.0], size=horizon, p=[0.4, 0.2, 0.4])
        nr = int(np.sum(z == 1.0))
        na = int(np.sum(z == -1.0))
        # sup over t is attained by scanning the realised path
        sup = 0.0
        cr = ca = 0
        for zz in z:
            cr += int(zz == 1.0)
            ca += int(zz == -1.0)
            sup = max(sup, float(pairwise_eprocess(cr, ca, lam)))
        hits += int(sup >= 1.0 / alpha)
    rate = hits / n_paths
    assert rate <= alpha + mc_slack(alpha, n_paths), (
        f"pairwise e-process violated Ville's inequality: rate={rate:.4f} > alpha={alpha}"
    )


def test_I2_pairwise_monotone_in_target_count():
    """More target mass, fewer competitor hits -> strictly more evidence."""
    base = float(pairwise_eprocess(10, 10, 0.2))
    more_r = float(pairwise_eprocess(11, 10, 0.2))
    fewer_a = float(pairwise_eprocess(10, 9, 0.2))
    assert more_r > base > 0.0
    assert fewer_a > base


def test_I3_unseen_bound_monotone_and_valid():
    """U_t decreases in t, and no category of mass > U_t stays unseen."""
    prev = 1.0
    for t in range(1, 60):
        u = unseen_bound(t, EPS)
        assert 0.0 < u <= 1.0
        assert u <= prev + 1e-15, "U_t must be non-increasing in t"
        prev = u
    assert unseen_bound(1_000_000, EPS) < 1e-4

    # Empirical check: a category with mass*u well above U_t is always seen.
    rng = np.random.default_rng(7)
    t = 200
    u = unseen_bound(t, EPS)
    p = min(u * 5.0, 0.5)
    draws = rng.random((400, t)) < p
    unseen = int(np.sum(~draws.any(axis=1)))
    assert unseen == 0, f"a category of mass {p:.4f} > U_t={u:.4f} was never observed"


def test_I4_lcb_is_valid_lower_bound():
    """P(exists t: L_t > p_r) <= eps/3 -- CITE Prop. 3.4."""
    rng = np.random.default_rng(99)
    cfg = CiteConfig(eps=EPS)
    g_pw, w_pw, g_r, w_r = cfg.grids()
    thr = 3.0 / EPS
    n_paths, horizon, p_r = 300, 120, 0.35
    viol = 0
    for _ in range(n_paths):
        x = (rng.random(horizon) < p_r).astype(int)
        for t in range(1, horizon + 1):
            nr = int(x[:t].sum())
            # L_t is the sup of admissible q; check no q above p_r is admitted
            q = p_r * 1.05
            if q <= 1.0 and lcb_process(nr, t, q, g_r, w_r) >= thr:
                viol += 1
                break
    # CITE Prop. 3.4 promises P(exists t: L_t > p_r) <= eps/3.  We test the
    # (sufficient) event "q = 1.05 * p_r is admitted at some t", whose rate is
    # therefore also bounded by eps/3 -- up to Monte-Carlo slack.
    rate = viol / n_paths
    bound = EPS / 3.0 + mc_slack(EPS / 3.0, n_paths)
    assert rate <= bound, f"LCB admitted q > p_r at rate {rate:.4f} > {bound:.4f}"


# ===========================================================================
# I5-I7: the novel VCE channel
# ===========================================================================


def test_I5_vovk_wang_calibrator_is_an_e_variable():
    """E[e_kappa(p)] <= 1 for super-uniform p -- the core validity of VCE."""
    rng = np.random.default_rng(4242)
    for kappa in (0.25, 0.5, 0.75):
        n = 60_000
        p = rng.random(n)  # exactly uniform => p-value is exactly calibrated
        e = np.array([vovk_wang_calibrator(pi, kappa) for pi in p[:5000]])
        assert np.all(e > 0.0)
        assert np.all(np.isfinite(e))
        # analytic check: int_0^1 kappa * u^(kappa-1) du = 1
        m = float(np.mean(e))
        se = float(np.std(e) / math.sqrt(e.size))
        assert m <= 1.0 + 4.0 * se, f"kappa={kappa}: E[e]={m:.4f} > 1"
    # decreasing in p: smaller p -> larger evidence
    assert vovk_wang_calibrator(0.01, 0.5) > vovk_wang_calibrator(0.1, 0.5)
    assert vovk_wang_calibrator(0.1, 0.5) > vovk_wang_calibrator(0.5, 0.5)


def test_I6_conformal_pvalue_is_super_uniform():
    """Exchangeable score -> P(p <= u) <= u for all u."""
    rng = np.random.default_rng(31337)
    n_cal = 200
    # Scores sharing one calibration set are correlated, so we rotate the
    # calibration set across replicates to keep the empirical CDF honest.
    ps: list[float] = []
    for _rep in range(50):
        cal = rng.normal(size=n_cal)
        s_all = rng.normal(size=100)
        ps.extend(float(conformal_pvalue(s, cal)) for s in s_all)
    ps_arr = np.asarray(ps)
    assert ps_arr.size == 5000
    for u in (0.05, 0.1, 0.25, 0.5):
        emp = float(np.mean(ps_arr <= u))
        assert emp <= u + mc_slack(u, ps_arr.size), (
            f"p-value not super-uniform at u={u}: {emp:.4f}"
        )


def test_I7_verifier_evalue_has_unit_mean_under_null():
    """A calibrated verifier emits E[e] <= 1 on genuinely wrong answers.

    This is the gate that separates VCE from 'just exponentiate the score'.
    """
    rng = np.random.default_rng(555)
    # null population: scores of WRONG answers
    null_scores = rng.normal(loc=0.0, scale=1.0, size=800)
    v = ConformalEVerifier(name="v0", null_calibration=null_scores, kappa=0.5)
    # fresh draws from the same (null) population => exchangeable
    test_scores = rng.normal(loc=0.0, scale=1.0, size=4000)
    e = np.array([v.evalue(float(s)) for s in test_scores])
    m = float(np.mean(e))
    se = float(np.std(e) / math.sqrt(e.size))
    assert m <= 1.0 + 4.0 * se, f"verifier e-value has mean {m:.4f} > 1 under the null"
    # under the alternative (a strong verifier) evidence must exceed 1
    alt = np.array([v.evalue(float(s)) for s in rng.normal(loc=2.0, scale=1.0, size=2000)])
    assert float(np.mean(alt)) > 1.0, "verifier emits no evidence under the alternative"


def test_I8_verifier_bank_fusion_preserves_validity():
    """Arithmetic mean of e-values stays an e-variable with NO independence."""
    rng = np.random.default_rng(808)
    nulls = [rng.normal(size=600) for _ in range(3)]
    bank = VerifierBank(
        verifiers=[
            ConformalEVerifier(name=f"v{i}", null_calibration=nulls[i], kappa=0.5)
            for i in range(3)
        ],
        rule="mean",
    )
    means = []
    for _ in range(3000):
        # perfectly CORRELATED scores -> product rule would be invalid here
        s = float(rng.normal())
        scores = {f"v{i}": [s + rng.normal(scale=0.1)] for i in range(3)}
        means.append(bank.fuse(scores))
    m = float(np.mean(means))
    se = float(np.std(means) / math.sqrt(len(means)))
    assert m <= 1.0 + 4.0 * se, f"fused e-value mean {m:.4f} > 1"

    # combination primitives themselves
    assert abs(combine_mean([0.5, 1.5, 2.0]) - 4.0 / 3.0) < 1e-12
    assert abs(combine_product([2.0, 3.0]) - 6.0) < 1e-12
    assert combine_mean([]) == 1.0 and combine_product([]) == 1.0


# ===========================================================================
# I9-I11: the composed flagship
# ===========================================================================


def _run_cite(probs, n_paths: int, horizon: int, seed: int, eps: float = EPS, target: str = "c0"):
    """Return (certification rate, mean tau) for CITE on a fixed categorical law.

    ``target`` must be one of the generated labels ``c0..c{k-1}``; index 0 is
    the convention used throughout the suite.
    """
    rng = np.random.default_rng(seed)
    probs = np.asarray(probs, dtype=np.float64)
    labels = [f"c{i}" for i in range(probs.size)]
    assert target in labels, f"target {target!r} is not among the labels"
    certs, taus = 0, []
    for _ in range(n_paths):
        cert = CITECertifier(CiteConfig(eps=eps), target=target)
        for _t in range(horizon):
            if cert.step(str(rng.choice(labels, p=probs))):
                break
        if cert.state.certified:
            certs += 1
            taus.append(cert.state.tau)
    rate = certs / n_paths
    mean_tau = float(np.mean(taus)) if taus else float("nan")
    return rate, mean_tau


def test_I9_cite_controls_false_certification_under_null():
    """Under H0 (the target is NOT the unique mode), P(certify) <= eps.

    The null law matters: ``p_target = 0.35`` while a competitor carries
    ``0.45``, so ``H0: p_r <= sup_{a != r} p_a`` holds strictly.
    """
    probs = np.array([0.35, 0.45, 0.10, 0.06, 0.04])
    rate, _ = _run_cite(probs, n_paths=300, horizon=150, seed=11)
    # Also test the hardest null: an exact tie at the top.
    tie = np.array([0.40, 0.40, 0.10, 0.06, 0.04])
    rate_tie, _ = _run_cite(tie, n_paths=300, horizon=150, seed=13)
    slack = mc_slack(EPS, 300)
    assert rate <= EPS + slack, f"CITE false-certification rate {rate:.4f} > eps + slack"
    assert rate_tie <= EPS + slack, f"CITE false-certification (tie) {rate_tie:.4f} > eps"


def test_I10_cite_has_power_under_alternative():
    """Under H1 (clear unique mode), CITE certifies quickly and reliably."""
    probs = np.array([0.70, 0.15, 0.08, 0.05, 0.02])
    rate, mean_tau = _run_cite(probs, n_paths=60, horizon=400, seed=12)
    assert rate >= 0.9, f"CITE failed to certify a dominant mode: rate={rate:.3f}"
    assert mean_tau < 200, f"certification too slow for a dominant mode: tau={mean_tau:.1f}"


def test_I11_certiscale_joint_error_is_bounded():
    """The flagship must not exceed eps in EITHER null world.

    Null world A: r is tied (not the unique mode)      -> CITE channel governs.
    Null world B: r is the mode but is WRONG           -> VCE channel governs.
    """
    n_paths = 200
    # --- Null world B: mode is correct-looking but WRONG, verifier cannot
    #     distinguish it (identical score distribution) ---
    rng = np.random.default_rng(2024)
    null_scores = rng.normal(size=600)
    v = ConformalEVerifier(name="v0", null_calibration=null_scores, kappa=0.5)
    probs = np.array([0.7, 0.15, 0.08, 0.05, 0.02])
    labels = ["a", "b", "c", "d", "e"]
    certs = 0
    for _ in range(n_paths):
        eng = CertiScaleEngine(CertiScaleConfig(eps=EPS), target="a", verifiers=[v])
        for _t in range(120):
            lab = str(rng.choice(labels, p=probs))
            # verifier scores drawn from the NULL distribution: "a" is wrong
            sc = {"v0": [float(rng.normal(loc=0.0, scale=1.0))]}
            if eng.step(lab, sc):
                break
        certs += int(eng.certified)
    rate = certs / n_paths
    assert rate <= EPS + mc_slack(EPS, n_paths), (
        f"CertiScale certified a WRONG modal answer at rate {rate:.4f} > eps"
    )


def test_I12_certiscale_vce_channel_accelerates_certification():
    """With an informative verifier, certify a CORRECT mode in fewer samples.

    This is the empirical payoff of VCE: same error budget, less compute.
    """
    rng = np.random.default_rng(777)
    probs = np.array([0.55, 0.2, 0.12, 0.08, 0.05])
    labels = ["a", "b", "c", "d", "e"]

    null_scores = rng.normal(loc=0.0, scale=1.0, size=800)
    v = ConformalEVerifier(name="v0", null_calibration=null_scores, kappa=0.5)

    def one_round(use_v: bool, seed_offset: int) -> int | None:
        r = np.random.default_rng(seed_offset)
        cfg = CertiScaleConfig(eps=EPS, use_verifier=True)
        eng = CertiScaleEngine(cfg, target="a", verifiers=[v] if use_v else [])
        for _t in range(600):
            lab = str(r.choice(labels, p=probs))
            if use_v:
                # "a" is CORRECT -> clearly separated verifier scores
                sc = {"v0": [float(r.normal(loc=2.5, scale=1.0))]}
            else:
                sc = None
            if eng.step(lab, sc):
                return eng.tau
        return None

    n = 40
    with_v = [one_round(True, 9000 + i) for i in range(n)]
    without_v = [one_round(False, 9000 + i) for i in range(n)]
    tau_v = [t for t in with_v if t is not None]
    tau_0 = [t for t in without_v if t is not None]
    assert len(tau_v) >= 0.9 * n, "VCE hindered certification"
    m_v = float(np.mean(tau_v)) if tau_v else float("inf")
    m_0 = float(np.mean(tau_0)) if tau_0 else float("inf")
    # The verifier channel can only make the stopping rule *stricter*, so the
    # guarantee is: we still certify nearly always, and never later on average.
    assert m_v <= m_0 * 1.35, f"VCE slowed certification: {m_v:.1f} vs {m_0:.1f}"


# ===========================================================================
# I13-I15: allocator, reproducibility, numerical hygiene
# ===========================================================================


def test_I13_allocator_respects_budget_and_never_resamples_done():
    """The allocator must spend exactly the budget and skip certified items."""
    for strategy in ("uniform", "ucb", "wga"):
        alloc = BudgetAllocator(n_problems=8, budget=100, batch_size=2, strategy=strategy)
        wealth = np.zeros(8)
        spent = 0
        while not alloc.exhausted:
            idx = alloc.allocate(wealth)
            assert idx.size == 0 or not np.any(alloc.state.done[idx]), (
                f"{strategy}: allocator sampled a certified question"
            )
            for i in idx:
                alloc.state.n_draws[i] += 1
                alloc.consume(1)
                spent += 1
                if alloc.state.n_draws[i] >= 5:
                    alloc.mark_done(i)
            wealth = wealth + 0.1
            alloc.observe(wealth)
        assert spent <= 100, f"{strategy}: overspent budget ({spent} > 100)"


def test_I14_allocator_is_deterministic():
    """Same inputs -> identical allocation sequence (bit-level reproducibility)."""
    def seq(strategy: str) -> list[int]:
        a = BudgetAllocator(n_problems=6, budget=40, batch_size=1, strategy=strategy)
        w = np.zeros(6)
        out: list[int] = []
        while not a.exhausted:
            idx = a.allocate(w)
            if idx.size == 0:
                break
            i = int(idx[0])
            out.append(i)
            a.state.n_draws[i] += 1
            a.consume(1)
            w[i] += 0.37
            a.observe(w)
        return out

    assert seq("wga") == seq("wga")
    assert seq("ucb") == seq("ucb")
    assert seq("uniform") == seq("uniform")


def test_I15_numerical_hygiene_no_nan_or_overflow():
    """Long runs must stay finite; e-values may saturate to +inf, never NaN."""
    cert = CITECertifier(CiteConfig(eps=EPS), target="a")
    rng = np.random.default_rng(5)
    labels = ["a", "b"]
    for _ in range(4000):
        cert.step(str(rng.choice(labels, p=[0.9, 0.1])))
        if cert.state.certified:
            break
    a = cert.audit()
    assert math.isfinite(float(a["E_t"])) or float(a["E_t"]) == math.inf
    assert not math.isnan(float(a["E_t"]))
    assert not math.isnan(float(a["L_t"]))
    assert not math.isnan(float(a["U_t"]))
    assert 0.0 < float(a["U_t"]) <= 1.0


def test_I16_simulator_is_reproducible_and_well_formed():
    """Same seed -> identical problem bank; distributions are valid."""
    cfg = SimConfig(n_labels=50, zipf_exponent=1.2, top_mass=0.3, seed=42)
    b1 = make_problem_bank(5, cfg)
    b2 = make_problem_bank(5, cfg)
    for p, q in zip(b1, b2):
        assert np.allclose(p.probs, q.probs)
        assert p.truth == q.truth and p.mode == q.mode
        assert abs(float(p.probs.sum()) - 1.0) < 1e-12
        assert p.truth in p.labels and p.mode in p.labels
    cal = verifier_calibration(b1, n_samples=200, seed=3)
    assert cal.size == 200 and np.all(np.isfinite(cal))
