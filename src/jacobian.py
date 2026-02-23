"""Finite-difference Jacobian estimation for behavioral observables.

Doctrine reference: constants.DOCTRINE and constants.DOCTRINE_HASH.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from constants import DOCTRINE_HASH
from src.behavioral import BehavioralConfig, behavioral_map
from src.system import DelayedOpenExpandingSystem, SystemParameters
from src.utils import append_jsonl, ensure_dir


@dataclass(frozen=True)
class JacobianConfig:
    behavioral: BehavioralConfig
    Z0_seed: int
    Z0_size: int
    h_values: tuple[float, ...]
    seeds: tuple[int, ...]
    plateau_tol: float
    epsilon: float


def finite_difference_jacobian(base_params: SystemParameters, cfg: JacobianConfig) -> dict:
    h_values = tuple(sorted(cfg.h_values, reverse=True))
    if h_values != cfg.h_values:
        raise ValueError("h_values must be sorted descending")

    ensure_dir("results")
    evals_path = Path("results") / "B_evals.jsonl"
    if evals_path.exists():
        evals_path.unlink()

    rng = np.random.default_rng(cfg.Z0_seed)
    Z0 = rng.random((cfg.Z0_size, 2), dtype=np.float64)

    directions = ("theta", "kappa", "mu")
    eval_cache: dict[tuple[int, str, float], np.ndarray] = {}
    ulam_logs: list[dict] = []
    escape_logs: list[dict] = []

    eval_id = 0
    for seed in cfg.seeds:
        system0 = DelayedOpenExpandingSystem(base_params)
        out0 = behavioral_map(system0, cfg.behavioral, Z0, seed)
        eval_cache[(seed, "base", 0.0)] = out0["B"]
        append_jsonl(
            evals_path,
            {
                "eval_id": eval_id,
                "seed": seed,
                "point": "base",
                "h": 0.0,
                "B": out0["B"].tolist(),
                "system_hash": system0.identity_hash(),
                "doctrine_hash": DOCTRINE_HASH,
            },
        )
        eval_id += 1
        ulam_logs.append({"seed": seed, "point": "base", "h": 0.0, **out0["ulam"]})
        escape_logs.append({"seed": seed, "point": "base", "h": 0.0, **out0["escape"]})

        for h in h_values:
            for p in directions:
                plus_params = base_params.with_updates(**{p: getattr(base_params, p) + h})
                minus_params = base_params.with_updates(**{p: getattr(base_params, p) - h})
                sys_plus = DelayedOpenExpandingSystem(plus_params)
                sys_minus = DelayedOpenExpandingSystem(minus_params)
                out_plus = behavioral_map(sys_plus, cfg.behavioral, Z0, seed)
                out_minus = behavioral_map(sys_minus, cfg.behavioral, Z0, seed)
                eval_cache[(seed, f"{p}+", h)] = out_plus["B"]
                eval_cache[(seed, f"{p}-", h)] = out_minus["B"]
                for sign, system_obj, out in (("+", sys_plus, out_plus), ("-", sys_minus, out_minus)):
                    append_jsonl(
                        evals_path,
                        {
                            "eval_id": eval_id,
                            "seed": seed,
                            "point": f"{p}{sign}",
                            "h": float(h),
                            "B": out["B"].tolist(),
                            "system_hash": system_obj.identity_hash(),
                            "doctrine_hash": DOCTRINE_HASH,
                        },
                    )
                    eval_id += 1
                    ulam_logs.append({"seed": seed, "point": f"{p}{sign}", "h": float(h), **out["ulam"]})
                    escape_logs.append({"seed": seed, "point": f"{p}{sign}", "h": float(h), **out["escape"]})

    derivatives: dict[str, list[np.ndarray]] = {p: [] for p in directions}
    stats: dict[str, list[dict]] = {p: [] for p in directions}
    for p in directions:
        for h in h_values:
            seed_derivs = []
            for seed in cfg.seeds:
                d = (eval_cache[(seed, f"{p}+", h)] - eval_cache[(seed, f"{p}-", h)]) / (2.0 * h)
                seed_derivs.append(d)
            arr = np.array(seed_derivs, dtype=np.float64)
            derivatives[p].append(arr)
            stats[p].append(
                {
                    "h": float(h),
                    "mean": np.mean(arr, axis=0).tolist(),
                    "sd": np.std(arr, axis=0, ddof=1).tolist() if arr.shape[0] > 1 else [0.0, 0.0, 0.0],
                }
            )

    plateau_info: dict[str, dict] = {}
    chosen_h: dict[str, float] = {}
    chosen_idx: dict[str, int] = {}

    for p in directions:
        chosen = len(h_values) - 1
        checks = []
        for i in range(len(h_values) - 1):
            D1 = derivatives[p][i]
            D2 = derivatives[p][i + 1]
            rel = np.abs(D2 - D1) / np.maximum(np.abs(D1), cfg.epsilon)
            rel_ok = bool(np.all(np.mean(rel, axis=0) < cfg.plateau_tol))
            sign_ok = bool(np.all(np.sign(D1) == np.sign(D2)))
            checks.append(
                {
                    "i": i,
                    "h_current": float(h_values[i]),
                    "h_next": float(h_values[i + 1]),
                    "mean_rel_change": np.mean(rel, axis=0).tolist(),
                    "rel_ok": rel_ok,
                    "sign_consistent": sign_ok,
                }
            )
            if rel_ok and sign_ok:
                chosen = i + 1
                break
        chosen_idx[p] = chosen
        chosen_h[p] = float(h_values[chosen])
        plateau_info[p] = {
            "checks": checks,
            "chosen_index": int(chosen),
            "chosen_h": float(h_values[chosen]),
            "rule": "first consecutive pair with rel_change < plateau_tol and sign consistency across seeds",
        }

    jacobians = {}
    dets, conds = [], []
    singvals = []
    col_norm_dets = []
    for seed in cfg.seeds:
        J = np.zeros((3, 3), dtype=np.float64)
        for col, p in enumerate(directions):
            h = chosen_h[p]
            J[:, col] = (eval_cache[(seed, f"{p}+", h)] - eval_cache[(seed, f"{p}-", h)]) / (2.0 * h)
        np.save(Path("results") / f"jacobian_seed_{seed}.npy", J)
        jacobians[seed] = J
        dets.append(float(np.linalg.det(J)))
        svals = np.linalg.svd(J, compute_uv=False)
        singvals.append(svals)
        conds.append(float(np.linalg.cond(J)))
        Jn = J / np.maximum(np.linalg.norm(J, axis=0, keepdims=True), cfg.epsilon)
        col_norm_dets.append(float(np.linalg.det(Jn)))

    J_stack = np.stack(list(jacobians.values()), axis=0)
    J_mean = np.mean(J_stack, axis=0)
    J_sd = np.std(J_stack, axis=0, ddof=1) if J_stack.shape[0] > 1 else np.zeros_like(J_mean)

    def ci95(arr: np.ndarray) -> list[float]:
        return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]

    summary = {
        "J_mean": J_mean.tolist(),
        "J_sd": J_sd.tolist(),
        "det_distribution": dets,
        "det_ci95": ci95(np.array(dets, dtype=np.float64)),
        "singular_values_distribution": np.array(singvals, dtype=np.float64).tolist(),
        "singular_values_ci95": np.percentile(np.array(singvals), [2.5, 97.5], axis=0).tolist(),
        "cond_distribution": conds,
        "cond_mean": float(np.mean(conds)),
        "column_normalized_det_distribution": col_norm_dets,
        "column_normalized_det_ci95": ci95(np.array(col_norm_dets, dtype=np.float64)),
        "derivative_by_h": stats,
    }

    return {
        "summary": summary,
        "plateau_info": plateau_info,
        "ulam_logs": ulam_logs,
        "escape_logs": escape_logs,
        "meta": {"Z0_seed": cfg.Z0_seed, "replicate_seeds": list(cfg.seeds), "Z0_size": cfg.Z0_size},
    }
