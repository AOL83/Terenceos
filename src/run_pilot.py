"""CLI entry point for end-to-end pilot bundle generation.

Doctrine reference: constants.DOCTRINE and constants.DOCTRINE_HASH.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml

from src.behavioral import BehavioralConfig
from src.jacobian import JacobianConfig, finite_difference_jacobian
from src.report import BUNDLE_REQUIRED_FILES, write_bundle
from src.system import DelayedOpenExpandingSystem, SystemParameters
from src.ulam import build_ulam_matrix, compute_ulam_spectrum
from src.utils import ensure_dir, now_utc_stamp


def run_pilot(config_path: str) -> Path:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))

    base_params = SystemParameters(**cfg["base_parameters"])
    beh_cfg = BehavioralConfig(**cfg["behavioral"])
    jac_cfg = JacobianConfig(
        behavioral=beh_cfg,
        Z0_seed=int(cfg["jacobian"]["Z0_seed"]),
        Z0_size=int(cfg["jacobian"]["Z0_size"]),
        h_values=tuple(float(x) for x in cfg["jacobian"]["h_values"]),
        seeds=tuple(int(x) for x in cfg["jacobian"]["seeds"]),
        plateau_tol=float(cfg["jacobian"]["plateau_tol"]),
        epsilon=float(cfg["jacobian"]["epsilon"]),
    )

    out_base = ensure_dir(cfg.get("output_base", "results"))
    run_id = f"run_{now_utc_stamp()}"
    bundle_dir = ensure_dir(out_base / run_id)

    old_cwd = Path.cwd()
    os.chdir(bundle_dir)
    try:
        jacobian_result = finite_difference_jacobian(base_params, jac_cfg)
        jacobian_result["system_hash_base"] = DelayedOpenExpandingSystem(base_params).identity_hash()

        grid_ref_rows = []
        for M in cfg["grid_refinement"]["M_list"]:
            P = build_ulam_matrix(DelayedOpenExpandingSystem(base_params), int(M), int(beh_cfg.ulam_r))
            spec = compute_ulam_spectrum(P)
            print(f"[ulam-grid] M={M} n={P.shape[0]} nnz={P.nnz}")
            grid_ref_rows.append(
                {
                    "M": int(M),
                    "r": int(beh_cfg.ulam_r),
                    "lambda1": spec["lambda1"],
                    "rho_proxy": float(-__import__("math").log(max(abs(spec["lambda1"]), 1e-15))),
                }
            )

        write_bundle(".", run_id, jacobian_result, grid_ref_rows)
        for req in BUNDLE_REQUIRED_FILES:
            if not Path(req).exists():
                raise RuntimeError(f"Required file missing post-write: {req}")
    finally:
        os.chdir(old_cwd)

    return bundle_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TerenceOS pilot bundle pipeline")
    parser.add_argument("--config", default="configs/pilot.yaml", help="Path to YAML config")
    args = parser.parse_args()
    bundle = run_pilot(args.config)
    print(bundle)


if __name__ == "__main__":
    main()
