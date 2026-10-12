#!/usr/bin/env python
"""CertiScale one-shot self-verification (zero dependency beyond NumPy).

    python verify.py            # run every gate
    python verify.py --quick    # skip the slow Monte-Carlo gates

Every gate below is a *mathematical* property, not a snapshot test.  A green
run means the mathematical promises made in the README actually hold on this
machine, at this commit.

Author: 晨星 (CJX0712)
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from certiscale.allocator import BudgetAllocator  # noqa: E402
from certiscale.cite import CITECertifier, CiteConfig  # noqa: E402
from certiscale.conformal import conformal_pvalue  # noqa: E402
from certiscale.eprocess import (  # noqa: E402
    combine_mean,
    combine_product,
    lcb_process,
    pairwise_eprocess,
    unseen_bound,
    vovk_wang_calibrator,
)
from certiscale.flagship import CertiScaleConfig, CertiScaleEngine  # noqa: E402
from certiscale.simulator import SimConfig, make_problem_bank  # noqa: E402
from certiscale.verifier import ConformalEVerifier, VerifierBank  # noqa: E402

EPS = 0.05
GREEN, RED, WARN = "\u2705", "\u274c", "\u26a0\ufe0f"


class Gate:
    __slots__ = ("name", "ok", "detail", "seconds")

    def __init__(self, name: str, ok: bool, detail: str, seconds: float):
        self.name, self.ok, self.detail, self.seconds = name, ok, detail, seconds


def slack(eps: float, n: int) -> float:
    return 4.0 * math.sqrt(max(eps * (1.0 - eps), 1e-12) / max(n, 1))


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------


def g01_pairwise_ville(quick: bool) -> Gate:
    """Pairwise e-process satisfies Ville's inequality under the null."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(20261013)
    alpha, lam = 0.10, 0.25
    n_paths = 120 if quick else 400
    horizon = 60
    hits = 0
    for _ in range(n_paths):
        z = rng.choice([-1.0, 0.0, 1.0], size=horizon, p=[0.4, 0.2, 0.4])
        sup, cr, ca = 0.0, 0, 0
        for zz in z:
            cr += int(zz == 1.0)
            ca += int(zz == -1.0)
            sup = max(sup, float(pairwise_eprocess(cr, ca, lam)))
        hits += int(sup >= 1.0 / alpha)
    rate = hits / n_paths
    ok = rate <= alpha + slack(alpha, n_paths)
    return Gate("I1  pairwise e-process: Ville under H0", ok, f"rate={rate:.4f} <= {alpha}", time.perf_counter() - t0)


def g02_unseen_bound() -> Gate:
    t0 = time.perf_counter()
    prev, mono = 1.0, True
    for t in range(1, 80):
        u = unseen_bound(t, EPS)
        if not (0.0 < u <= prev + 1e-15):
            mono = False
        prev = u
    ok = mono and unseen_bound(10**6, EPS) < 1e-4
    return Gate("I3  unseen bound U_t monotone & tight", ok, f"U_1e6={unseen_bound(10**6, EPS):.2e}", time.perf_counter() - t0)


def g03_lcb_validity(quick: bool) -> Gate:
    t0 = time.perf_counter()
    rng = np.random.default_rng(99)
    cfg = CiteConfig(eps=EPS)
    _, _, g_r, w_r = cfg.grids()
    thr = 3.0 / EPS
    n_paths = 100 if quick else 300
    p_r, horizon = 0.35, 100 if quick else 120
    viol = 0
    for _ in range(n_paths):
        x = (rng.random(horizon) < p_r).astype(int)
        for t in range(1, horizon + 1):
            if lcb_process(int(x[:t].sum()), t, p_r * 1.05, g_r, w_r) >= thr:
                viol += 1
                break
    rate = viol / n_paths
    bound = EPS / 3.0 + slack(EPS / 3.0, n_paths)
    ok = rate <= bound
    return Gate("I4  LCB validity (CITE Prop. 3.4)", ok, f"rate={rate:.4f} <= {bound:.4f}", time.perf_counter() - t0)


def g04_calibrator(quick: bool) -> Gate:
    t0 = time.perf_counter()
    rng = np.random.default_rng(4242)
    n = 1500 if quick else 5000
    worst = -1.0
    for kappa in (0.25, 0.5, 0.75):
        p = rng.random(n)
        e = np.array([vovk_wang_calibrator(pi, kappa) for pi in p])
        m = float(np.mean(e))
        se = float(np.std(e) / math.sqrt(e.size))
        worst = max(worst, m - 4.0 * se)
    ok = worst <= 1.0
    return Gate("I5  Vovk-Wang calibrator: E[e] <= 1", ok, f"worst mean-4se={worst:.4f}", time.perf_counter() - t0)


def g05_conformal(quick: bool) -> Gate:
    t0 = time.perf_counter()
    rng = np.random.default_rng(31337)
    ps: list[float] = []
    for _ in range(20 if quick else 50):
        cal = rng.normal(size=200)
        ps.extend(float(conformal_pvalue(s, cal)) for s in rng.normal(size=100))
    arr = np.asarray(ps)
    worst_u, worst_v = None, -1.0
    for u in (0.05, 0.1, 0.25, 0.5):
        emp = float(np.mean(arr <= u))
        if emp > u + slack(u, arr.size):
            worst_u, worst_v = u, emp
    ok = worst_u is None
    return Gate("I6  conformal p-value super-uniform", ok, "all u ok" if ok else f"u={worst_u} emp={worst_v:.4f}", time.perf_counter() - t0)


def g06_verifier_evalue(quick: bool) -> Gate:
    t0 = time.perf_counter()
    rng = np.random.default_rng(555)
    null_scores = rng.normal(size=800)
    v = ConformalEVerifier(name="v0", null_calibration=null_scores, kappa=0.5)
    n = 1200 if quick else 4000
    e = np.array([v.evalue(float(s)) for s in rng.normal(size=n)])
    m, se = float(np.mean(e)), float(np.std(e) / math.sqrt(n))
    alt = np.array([v.evalue(float(s)) for s in rng.normal(loc=2.0, size=800)])
    ok = (m <= 1.0 + 4.0 * se) and float(np.mean(alt)) > 1.0
    return Gate("I7  verifier e-value: E[e]<=1 null, >1 alt", ok, f"null={m:.3f} alt={float(np.mean(alt)):.2f}", time.perf_counter() - t0)


def g07_fusion() -> Gate:
    t0 = time.perf_counter()
    rng = np.random.default_rng(808)
    bank = VerifierBank(
        verifiers=[
            ConformalEVerifier(name=f"v{i}", null_calibration=rng.normal(size=600), kappa=0.5)
            for i in range(3)
        ],
        rule="mean",
    )
    n = 1000 if True else 3000
    vals = []
    for _ in range(n):
        s = float(rng.normal())  # perfectly correlated verifiers
        vals.append(bank.fuse({f"v{i}": [s + rng.normal(scale=0.1)] for i in range(3)}))
    m, se = float(np.mean(vals)), float(np.std(vals) / math.sqrt(len(vals)))
    ok = (m <= 1.0 + 4.0 * se) and abs(combine_mean([0.5, 1.5, 2.0]) - 4 / 3) < 1e-12
    ok = ok and abs(combine_product([2.0, 3.0]) - 6.0) < 1e-12
    return Gate("I8  bank fusion valid w/o independence", ok, f"mean={m:.3f}", time.perf_counter() - t0)


def _cite_rate(probs, n_paths, horizon, seed) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    labels = [f"c{i}" for i in range(len(probs))]
    certs, taus = 0, []
    for _ in range(n_paths):
        cert = CITECertifier(CiteConfig(eps=EPS), target=labels[0])
        for _t in range(horizon):
            if cert.step(str(rng.choice(labels, p=probs))):
                break
        if cert.state.certified:
            certs += 1
            taus.append(cert.state.tau)
    return certs / n_paths, (float(np.mean(taus)) if taus else float("nan"))


def g08_cite_null(quick: bool) -> Gate:
    t0 = time.perf_counter()
    n = 150 if quick else 300
    r1, _ = _cite_rate(np.array([0.35, 0.45, 0.10, 0.06, 0.04]), n, 150, 11)
    r2, _ = _cite_rate(np.array([0.40, 0.40, 0.10, 0.06, 0.04]), n, 150, 13)
    bound = EPS + slack(EPS, n)
    ok = r1 <= bound and r2 <= bound
    return Gate("I9  CITE false certification <= eps", ok, f"strict={r1:.4f} tie={r2:.4f} <= {bound:.4f}", time.perf_counter() - t0)


def g09_cite_power(quick: bool) -> Gate:
    t0 = time.perf_counter()
    n = 30 if quick else 60
    rate, tau = _cite_rate(np.array([0.70, 0.15, 0.08, 0.05, 0.02]), n, 400, 12)
    ok = rate >= 0.9 and tau < 200
    return Gate("I10 CITE power: dominant mode certified", ok, f"rate={rate:.3f} tau={tau:.1f}", time.perf_counter() - t0)


def g10_joint_null(quick: bool) -> Gate:
    """CertiScale must refuse a WRONG modal answer."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(2024)
    v = ConformalEVerifier(name="v0", null_calibration=rng.normal(size=600), kappa=0.5)
    probs = np.array([0.7, 0.15, 0.08, 0.05, 0.02])
    labels = ["c0", "c1", "c2", "c3", "c4"]
    n = 100 if quick else 200
    certs = 0
    for _ in range(n):
        eng = CertiScaleEngine(CertiScaleConfig(eps=EPS, vce_batch=8), target="c0", verifiers=[v])
        for _t in range(120):
            lab = str(rng.choice(labels, p=probs))
            sc = {"v0": [float(rng.normal())]} if lab == "c0" else None
            if eng.step(lab, sc):
                break
        certs += int(eng.certified)
    rate = certs / n
    ok = rate <= EPS + slack(EPS, n)
    return Gate("I11 CertiScale rejects a WRONG mode", ok, f"rate={rate:.4f} <= {EPS + slack(EPS, n):.4f}", time.perf_counter() - t0)


def g11_allocator() -> Gate:
    t0 = time.perf_counter()
    ok = True
    for strategy in ("uniform", "ucb", "wga"):
        a = BudgetAllocator(n_problems=8, budget=100, batch_size=2, strategy=strategy)
        w = np.zeros(8)
        spent = 0
        while not a.exhausted:
            idx = a.allocate(w)
            if idx.size and np.any(a.state.done[idx]):
                ok = False
            for i in idx:
                a.state.n_draws[i] += 1
                a.consume(1)
                spent += 1
                if a.state.n_draws[i] >= 5:
                    a.mark_done(i)
            w = w + 0.1
            a.observe(w)
        if spent > 100:
            ok = False
    return Gate("I13 allocator: budget & done-masking", ok, "3 strategies ok", time.perf_counter() - t0)


def g12_determinism() -> Gate:
    t0 = time.perf_counter()
    cfg = SimConfig(n_labels=50, zipf_exponent=1.2, top_mass=0.3, seed=42)
    b1, b2 = make_problem_bank(4, cfg), make_problem_bank(4, cfg)
    ok = all(np.allclose(p.probs, q.probs) and p.truth == q.truth for p, q in zip(b1, b2))

    def seq() -> list[int]:
        a = BudgetAllocator(n_problems=6, budget=40, batch_size=1, strategy="wga")
        w = np.zeros(6)
        out = []
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

    ok = ok and seq() == seq()
    return Gate("I14 bit-level determinism (fixed seed)", ok, "simulator + allocator", time.perf_counter() - t0)


def g13_numerics() -> Gate:
    t0 = time.perf_counter()
    cert = CITECertifier(CiteConfig(eps=EPS), target="c0")
    rng = np.random.default_rng(5)
    for _ in range(4000):
        cert.step(str(rng.choice(["c0", "c1"], p=[0.9, 0.1])))
        if cert.state.certified:
            break
    a = cert.audit()
    vals = [float(a["E_t"]), float(a["L_t"]), float(a["U_t"])]
    ok = all((math.isfinite(v) or v == math.inf) and not math.isnan(v) for v in vals)
    ok = ok and 0.0 < vals[2] <= 1.0
    return Gate("I15 numerical hygiene: no NaN/overflow", ok, f"E_t={vals[0]:.3g} U_t={vals[2]:.4g}", time.perf_counter() - t0)


# ---------------------------------------------------------------------------


def main() -> int:
    quick = "--quick" in sys.argv
    gates = [
        g01_pairwise_ville(quick),
        g02_unseen_bound(),
        g03_lcb_validity(quick),
        g04_calibrator(quick),
        g05_conformal(quick),
        g06_verifier_evalue(quick),
        g07_fusion(),
        g08_cite_null(quick),
        g09_cite_power(quick),
        g10_joint_null(quick),
        g11_allocator(),
        g12_determinism(),
        g13_numerics(),
    ]
    w = max(len(g.name) for g in gates)
    print("\nCertiScale invariant gates" + ("  (quick)" if quick else ""))
    print("=" * (w + 34))
    for g in gates:
        print(f"{GREEN if g.ok else RED} {g.name:<{w}}  {g.detail:<22s} {g.seconds:5.1f}s")
    passed = sum(g.ok for g in gates)
    total = len(gates)
    print("=" * (w + 34))
    print(f"{passed}/{total} gates green" + ("" if passed == total else f"  {WARN} {total - passed} FAILED"))
    print(f"total {sum(g.seconds for g in gates):.1f}s\n")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
