"""Behavioral observable map B(theta,kappa,mu).

Doctrine reference: constants.DOCTRINE.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.escape import estimate_escape_rate
from src.lyapunov import lyapunov_conditioned_split
from src.system import DelayedOpenExpandingSystem
from src.ulam import build_ulam_matrix, compute_ulam_spectrum


@dataclass(frozen=True)
class BehavioralConfig:
    T_lyap: int
    burn_lyap: int
    T_escape: int
    min_survivors: int
    ulam_M: int
    ulam_r: int


def behavioral_map(
    system: DelayedOpenExpandingSystem,
    cfg: BehavioralConfig,
    Z0: np.ndarray,
    seed: int,
) -> dict:
    lyap = lyapunov_conditioned_split(system, Z0, cfg.T_lyap, cfg.burn_lyap, seed)
    esc = estimate_escape_rate(system, Z0, cfg.T_escape, cfg.min_survivors)
    P = build_ulam_matrix(system, cfg.ulam_M, cfg.ulam_r)
    ulam_spec = compute_ulam_spectrum(P)
    lam2 = complex(ulam_spec["lambda2"][0], ulam_spec["lambda2"][1])

    B1 = float(lyap["lyap"])
    B2 = float(esc["rho"])
    B3 = float(-np.log(np.abs(lam2)))
    return {"B": np.array([B1, B2, B3], dtype=np.float64), "lyap": lyap, "escape": esc, "ulam": ulam_spec}
