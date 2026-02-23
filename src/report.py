"""Bundle writer and validator for pilot outputs.

Doctrine reference: constants.DOCTRINE and constants.DOCTRINE_HASH.
"""

from __future__ import annotations

import csv
import io
import json
import platform
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from constants import DOCTRINE, DOCTRINE_HASH, TIME_UNIT_DESCRIPTION
from src.utils import ensure_dir, get_git_commit, write_json

BUNDLE_REQUIRED_FILES = [
    "meta.json",
    "doctrine.txt",
    "jacobian_summary.json",
    "plateau_info.json",
    "ulam_logs.json",
    "escape_logs.json",
    "tables/grid_refinement.csv",
    "plots/plateau_theta.png",
    "plots/plateau_kappa.png",
    "plots/plateau_mu.png",
    "plots/survival_fit.png",
]


def _numpy_config_text() -> str:
    try:
        cfg = np.__config__.show(mode="dicts")
        return json.dumps(cfg, sort_keys=True)
    except Exception:
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                np.__config__.show()
            return buf.getvalue()
        except Exception as exc:
            return f"numpy_config_unavailable: {exc}"


def _plot_plateau(path: Path, param: str, derivative_by_h: list[dict], chosen_h: float) -> None:
    h = [d["h"] for d in derivative_by_h]
    means = np.array([d["mean"] for d in derivative_by_h], dtype=np.float64)
    plt.figure(figsize=(6, 4))
    for i, label in enumerate(["dB1", "dB2", "dB3"]):
        plt.plot(h, means[:, i], marker="o", label=label)
    plt.axvline(chosen_h, color="black", linestyle="--", label=f"chosen h={chosen_h:g}")
    plt.xscale("log")
    plt.gca().invert_xaxis()
    plt.xlabel("h")
    plt.ylabel("central difference")
    plt.title(f"Plateau diagnostics for {param}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_survival(path: Path, escape_logs: list[dict]) -> None:
    base = next(x for x in escape_logs if x["point"] == "base")
    surv = np.array(base["survivors"], dtype=np.float64)
    t = np.arange(len(surv), dtype=np.float64)
    plt.figure(figsize=(6, 4))
    plt.plot(t, np.log(np.clip(surv, 1.0, None)), label="log survivors")
    a, b = base["fit_window"]
    tt = np.arange(a, b + 1, dtype=np.float64)
    fit = base["slope"] * tt + base["intercept"]
    plt.plot(tt, fit, "--", label="fit window")
    plt.xlabel("t")
    plt.ylabel("log survivors")
    plt.title("Escape-rate fit window")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def write_bundle(
    bundle_dir: str | Path,
    run_id: str,
    jacobian_result: dict,
    grid_refinement_rows: list[dict],
    doctrine_file: str = "constants.py",
) -> Path:
    bundle = ensure_dir(bundle_dir)
    ensure_dir(bundle / "tables")
    ensure_dir(bundle / "plots")

    meta = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy_config": _numpy_config_text(),
        "seeds": {
            "Z0_seed": jacobian_result["meta"]["Z0_seed"],
            "replicate_seeds": jacobian_result["meta"]["replicate_seeds"],
        },
        "doctrine_hash": DOCTRINE_HASH,
        "doctrine_file": doctrine_file,
        "system_hash_base": jacobian_result["system_hash_base"],
        "time_unit": TIME_UNIT_DESCRIPTION,
    }
    write_json(bundle / "meta.json", meta)
    (bundle / "doctrine.txt").write_text(DOCTRINE, encoding="utf-8")
    write_json(bundle / "jacobian_summary.json", jacobian_result["summary"])
    write_json(bundle / "plateau_info.json", jacobian_result["plateau_info"])
    write_json(bundle / "ulam_logs.json", jacobian_result["ulam_logs"])
    write_json(bundle / "escape_logs.json", jacobian_result["escape_logs"])

    with (bundle / "tables/grid_refinement.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["M", "r", "lambda1", "rho_proxy"])
        writer.writeheader()
        for row in grid_refinement_rows:
            writer.writerow(row)

    deriv = jacobian_result["summary"]["derivative_by_h"]
    pi = jacobian_result["plateau_info"]
    _plot_plateau(bundle / "plots/plateau_theta.png", "theta", deriv["theta"], pi["theta"]["chosen_h"])
    _plot_plateau(bundle / "plots/plateau_kappa.png", "kappa", deriv["kappa"], pi["kappa"]["chosen_h"])
    _plot_plateau(bundle / "plots/plateau_mu.png", "mu", deriv["mu"], pi["mu"]["chosen_h"])
    _plot_survival(bundle / "plots/survival_fit.png", jacobian_result["escape_logs"])

    missing = [f for f in BUNDLE_REQUIRED_FILES if not (bundle / f).exists()]
    if missing:
        raise RuntimeError(f"Bundle incomplete; missing files: {missing}")
    return bundle
