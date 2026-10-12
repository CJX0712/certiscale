"""CertiScale (衡证) -- certifiable test-time compute scaling.

World-class composition, zero reinvention
----------------------------------------
Every primitive in this package is a known-optimal construction from the
literature; the contribution of this repository is their **composition** into
the first anytime-valid certification procedure that uses verifier evidence
and a shared compute budget while keeping the Type-I error *exactly* bounded.

    CITE   Ota et al. (2026), arXiv:2605.05873
           anytime-valid mode certification on an unknown/infinite alphabet
           -> ``cite.py``          (Tier-0 baseline, reproduced verbatim)

    Vovk & Wang (2021), Annals of Statistics 49(3)
           p-to-e calibration: e_kappa(p) = kappa * p^(kappa-1)
           -> ``eprocess.py``      (primary contribution of the VCE channel)

    Vovk, Gammerman & Shafer (2005)
           conformal p-values (distribution-free, exchangeability only)
           -> ``conformal.py``

    Ramdas, Ruf, Larsson & Koolen (2023), arXiv:2209.11119
           testing-by-betting / e-processes
           -> ``eprocess.py``

    Karnin, Koren & Somekh (2013) / Audibert & Bubeck (2010)
           sequential halving / budgeted pure exploration
           -> ``allocator.py``     (AVBA: allocation driven by log-wealth)

Novel components contributed here
---------------------------------
    VCE   Verifier-Calibrated E-process   ``verifier.py``
          turns ANY black-box verifier into a valid e-process
    AVBA  Anytime-Valid Budget Allocation ``allocator.py``
          allocates a shared sampling budget by log-wealth growth
    CertiScale                            ``flagship.py``
          CITE x VCE, error bounded by max(eps_cite, eps_vce), not the sum

Author: 晨星 (CJX0712)
"""

from __future__ import annotations

from .cite import CITECertifier, CiteConfig
from .eprocess import (
    combine_mean,
    combine_product,
    lcb_process,
    pairwise_eprocess,
    unseen_bound,
    vovk_wang_calibrator,
)
from .flagship import CertiScaleConfig, CertiScaleEngine, CertiScaleSystem
from .verifier import ConformalEVerifier, VerifierBank

__version__ = "0.1.0"
__author__ = "晨星 (CJX0712)"

__all__ = [
    "__version__",
    "__author__",
    "CITECertifier",
    "CiteConfig",
    "CertiScaleConfig",
    "CertiScaleEngine",
    "CertiScaleSystem",
    "ConformalEVerifier",
    "VerifierBank",
    "pairwise_eprocess",
    "lcb_process",
    "unseen_bound",
    "vovk_wang_calibrator",
    "combine_mean",
    "combine_product",
]
