"""CertiScale benchmark -- head-to-head against the CITE baseline.

What is being measured, and why it is the right thing to measure
---------------------------------------------------------------
Adding the VCE channel makes the stopping rule **stricter** (both channels must
fire), so it cannot certify *sooner*.  Anyone claiming "our method is faster"
here would be selling something.  The honest trade-off is:

    CITE        certifies the *modal* answer.  When the model is consistently
                wrong, CITE certifies the wrong answer just as confidently.
    CertiScale  certifies the modal answer **only if the verifier agrees**, and
                pays for that in a modest number of extra samples.

So the headline metric is the **false discovery rate among certified answers**
(FDR), reported next to its cost in samples.  A method that never certifies
would trivially achieve FDR = 0, so power is reported alongside and every
regime also reports the sample cost.  Three regimes are covered, including the
diffuse-tail setting that CITE singles out as the hard one.

Reuse note: the trap construction below (mode != truth) is the standard way to
probe "confidently wrong" behaviour; the Zipf regimes mirror CITE Sec. 6.

Usage
-----
    python bench/run_bench.py --seeds 5 --out bench/results/bench.json

Author: 晨星 (CJX0712)
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

import numpy as np

from certiscale.cite import CITECertifier, CiteConfig
from certiscale.flagship import CertiScaleConfig, CertiScaleEngine
from certiscale.simulator import SimConfig, make_problem_bank, verifier_calibration
from certiscale.verifier import ConformalEVerifier

EPS = 0.05


# ---------------------------------------------------------------------------
# regimes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Regime:
    name: str
    n_labels: int
    zipf_exponent: float
    top_mass: float
    verifier_sep: float
    trap_fraction: float  # share of questions whose MODE is WRONG
    horizon: int


REGIMES: list[Regime] = [
    Regime("moderate", 100, 1.2, 0.35, 1.5, 0.30, 300),
    Regime("diffuse_tail", 500, 1.1, 0.25, 1.5, 0.30, 400),
    Regime("near_tie", 60, 1.3, 0.30, 1.0, 0.30, 400),
]


# ---------------------------------------------------------------------------
# one experimental cell
# ---------------------------------------------------------------------------


@dataclass
class CellResult:
    regime: str
    method: str
    seed: int
    n_problems: int
    n_certified: int
    n_certified_correct: int
    power: float          # certified & correct / all correct-mode questions
    fdr: float            # certified & wrong / certified
    mean_tau: float       # mean samples spent on certified questions
    mean_tau_all: float   # mean samples spent overall (incl. non-certified)
    wall_s: float


def build_bank(regime: Regime, n_problems: int, seed: int):
    """Build a question bank where ``trap_fraction`` of questions are traps.

    A *trap* is a question whose modal answer is **wrong**: exactly the
    situation in which CITE (which only looks at the mode) is confidently
    wrong and CertiScale (which also looks at the verifier) should abstain.
    """
    rng = np.random.default_rng(seed)
    n_trap = int(round(regime.trap_fraction * n_problems))
    honest = make_problem_bank(
        n_problems - n_trap,
        SimConfig(
            n_labels=regime.n_labels,
            zipf_exponent=regime.zipf_exponent,
            top_mass=regime.top_mass,
            truth_is_mode=True,
            verifier_sep=regime.verifier_sep,
            n_verifiers=1,
            seed=seed + 1,
        ),
    )
    traps = make_problem_bank(
        n_trap,
        SimConfig(
            n_labels=regime.n_labels,
            zipf_exponent=regime.zipf_exponent,
            top_mass=regime.top_mass,
            truth_is_mode=False,
            verifier_sep=regime.verifier_sep,
            n_verifiers=1,
            seed=seed + 2,
        ),
    )
    bank = honest + traps
    rng.shuffle(bank)  # type: ignore[argparse-type]
    return bank


def run_cell(
    regime: Regime,
    method: str,
    seed: int,
    n_problems: int = 40,
    eps: float = EPS,
    vce_batch: int = 8,
) -> CellResult:
    """Run one (regime, method, seed) cell."""
    t0 = time.perf_counter()
    bank = build_bank(regime, n_problems, seed)

    # Calibration is built from HELD-OUT wrong answers -- never from the
    # questions under test, which is what keeps the conformal step honest.
    # The batch size MUST match the engine's, or the conformal statistic is
    # compared against the wrong reference distribution.
    cal = verifier_calibration(
        bank, n_samples=600, seed=seed + 99, verifier_index=0, batch=vce_batch
    )
    verifier = ConformalEVerifier(name="v0", null_calibration=cal, kappa=0.5)

    n_cert = 0
    n_cert_correct = 0
    taus: list[int] = []
    all_taus: list[int] = []
    n_correct_mode = sum(1 for p in bank if p.truth_is_mode)

    for p in bank:
        target = p.labels[int(np.argmax(p.probs))]  # the MODE (what we certify)
        used = 0
        certified = False
        if method == "cite":
            cert = CITECertifier(CiteConfig(eps=eps), target=target)
            for _ in range(regime.horizon):
                lab = p.sample_label()
                used += 1
                if cert.step(lab):
                    certified = True
                    break
        elif method == "certiscale":
            eng = CertiScaleEngine(
                CertiScaleConfig(eps=eps, vce_batch=vce_batch),
                target=target,
                verifiers=[verifier],
            )
            for _ in range(regime.horizon):
                lab, sc = p.sample()
                used += 1
                scores = {"v0": [sc["v0"]]} if lab == target else None
                if eng.step(lab, scores):
                    certified = True
                    break
        else:
            raise ValueError(f"unknown method: {method!r}")

        all_taus.append(used)
        if certified:
            n_cert += 1
            taus.append(used)
            # trap questions have mode != truth; certifying the mode is a wrong
            # discovery.  Honest questions: certifying the mode is correct.
            if p.truth_is_mode:
                n_cert_correct += 1

    wall = time.perf_counter() - t0
    return CellResult(
        regime=regime.name,
        method=method,
        seed=seed,
        n_problems=n_problems,
        n_certified=n_cert,
        n_certified_correct=n_cert_correct,
        power=(n_cert_correct / n_correct_mode) if n_correct_mode else float("nan"),
        fdr=((n_cert - n_cert_correct) / n_cert) if n_cert else 0.0,
        mean_tau=float(np.mean(taus)) if taus else float("nan"),
        mean_tau_all=float(np.mean(all_taus)) if all_taus else float("nan"),
        wall_s=wall,
    )


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------


def aggregate(cells: list[CellResult]) -> dict[str, dict[str, float]]:
    """Mean +/- standard error across seeds, per (regime, method)."""
    out: dict[str, dict[str, float]] = {}
    keys = [(c.regime, c.method) for c in cells]
    for regime, method in sorted(set(keys)):
        sub = [c for c in cells if c.regime == regime and c.method == method]
        def ms(field: str) -> tuple[float, float]:
            vals = np.array([getattr(c, field) for c in sub], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                return float("nan"), float("nan")
            m = float(np.mean(vals))
            se = float(np.std(vals, ddof=1) / math.sqrt(vals.size)) if vals.size > 1 else 0.0
            return m, se
        entry: dict[str, float] = {}
        for f in ("power", "fdr", "mean_tau", "mean_tau_all", "n_certified", "wall_s"):
            m, se = ms(f)
            entry[f"{f}_mean"] = m
            entry[f"{f}_se"] = se
        out[f"{regime}|{method}"] = entry
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="CertiScale benchmark")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--problems", type=int, default=40)
    ap.add_argument("--eps", type=float, default=EPS)
    ap.add_argument("--methods", type=str, default="cite,certiscale")
    ap.add_argument("--vce-batch", type=int, default=8)
    ap.add_argument("--out", type=str, default="bench/results/bench.json")
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    cells: list[CellResult] = []
    for regime in REGIMES:
        for method in methods:
            for s in range(args.seeds):
                c = run_cell(
                    regime,
                    method,
                    seed=1000 + 17 * s,
                    n_problems=args.problems,
                    eps=args.eps,
                    vce_batch=args.vce_batch,
                )
                cells.append(c)
                print(
                    f"[{regime.name:12s}] {method:11s} seed={c.seed} "
                    f"cert={c.n_certified:3d} power={c.power:.3f} "
                    f"fdr={c.fdr:.3f} tau={c.mean_tau:.1f}"
                )

    agg = aggregate(cells)
    payload = {
        "meta": {
            "eps": args.eps,
            "seeds": args.seeds,
            "problems_per_cell": args.problems,
            "methods": methods,
            "regimes": [asdict(r) for r in REGIMES],
            "generated_by": "bench/run_bench.py",
            "author": "晨星 (CJX0712)",
        },
        "aggregate": agg,
        "cells": [asdict(c) for c in cells],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out_path}")

    # headline comparison
    print("\n=== aggregate (mean +/- se over seeds) ===")
    for regime in REGIMES:
        row_c = agg.get(f"{regime.name}|cite")
        row_s = agg.get(f"{regime.name}|certiscale")
        if row_c and row_s:
            print(
                f"{regime.name:12s}  CITE       FDR={row_c['fdr_mean']:.3f}+/-{row_c['fdr_se']:.3f}"
                f"  power={row_c['power_mean']:.3f}  tau={row_c['mean_tau_mean']:.1f}"
            )
            print(
                f"{'':12s}  CertiScale FDR={row_s['fdr_mean']:.3f}+/-{row_s['fdr_se']:.3f}"
                f"  power={row_s['power_mean']:.3f}  tau={row_s['mean_tau_mean']:.1f}"
            )


if __name__ == "__main__":
    main()
