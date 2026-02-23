"""Escape-rate estimation aligned with the Ulam operator convention.

Doctrine reference: constants.DOCTRINE governs escape timing and time units.
"""

from __future__ import annotations

import numpy as np

from src.system import DelayedOpenExpandingSystem


def _fit_line(t: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    A = np.vstack([t, np.ones_like(t)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    y_hat = slope * t + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(slope), float(intercept), float(r2)


def _default_window(
    survivors: np.ndarray,
    min_survivors: int,
    r2_threshold: float,
    min_window_len: int,
) -> tuple[tuple[int, int], float, float, float, int]:
    valid = survivors >= min_survivors
    n = len(survivors)
    best: tuple[int, int] | None = None
    best_fit: tuple[float, float, float] | None = None
    candidates = 0

    i = 0
    while i < n:
        if not valid[i]:
            i += 1
            continue
        j = i
        while j < n and valid[j]:
            j += 1
        block_start, block_end = i, j - 1
        for a in range(block_start, block_end + 1):
            for b in range(a + min_window_len - 1, block_end + 1):
                candidates += 1
                y = np.log(np.clip(survivors[a : b + 1].astype(np.float64), 1.0, None))
                t = np.arange(a, b + 1, dtype=np.float64)
                slope, intercept, r2 = _fit_line(t, y)
                if r2 < r2_threshold:
                    continue
                if best is None or (b - a) > (best[1] - best[0]):
                    best = (a, b)
                    best_fit = (slope, intercept, r2)
        i = j

    if best is None or best_fit is None:
        raise RuntimeError("No valid default fit window found under policy constraints")
    return best, best_fit[0], best_fit[1], best_fit[2], candidates


def estimate_escape_rate(
    system: DelayedOpenExpandingSystem,
    Z0: np.ndarray,
    T: int,
    min_survivors: int,
    window: tuple[int, int] | None = None,
    r2_threshold: float = 0.98,
    min_window_len: int = 25,
    fallback_policy: str = "strict_fail",
) -> dict:
    states = np.asarray(Z0, dtype=np.float64).copy()
    survivors = [int(states.shape[0])]

    for _ in range(T):
        x_next, y_next = system.step(states[:, 0], states[:, 1])
        keep = ~np.asarray(system.escaped_new_x(x_next), dtype=bool)
        states = np.column_stack((x_next[keep], y_next[keep])).astype(np.float64)
        survivors.append(int(states.shape[0]))
        if states.shape[0] == 0:
            break

    survivors_arr = np.asarray(survivors, dtype=np.int64)
    if window is None:
        try:
            fit_window, slope, intercept, r2, cand = _default_window(
                survivors_arr,
                min_survivors,
                r2_threshold,
                min_window_len,
            )
            fallback_used = False
            policy_name = "largest_contiguous_segment"
        except RuntimeError:
            if fallback_policy == "strict_fail":
                raise
            if fallback_policy != "best_available":
                raise ValueError("fallback_policy must be one of {'strict_fail', 'best_available'}")

            fallback_used = True
            policy_name = "best_available"
            cand = 0
            best: tuple[int, int] | None = None
            best_fit: tuple[float, float, float] | None = None
            fallback_min_len = max(10, min_window_len // 2)
            n = len(survivors_arr)
            for a in range(n):
                for b in range(a + fallback_min_len - 1, n):
                    if np.min(survivors_arr[a : b + 1]) < 1:
                        continue
                    cand += 1
                    t = np.arange(a, b + 1, dtype=np.float64)
                    y = np.log(np.clip(survivors_arr[a : b + 1].astype(np.float64), 1.0, None))
                    slope_i, intercept_i, r2_i = _fit_line(t, y)
                    if best_fit is None or r2_i > best_fit[2]:
                        best = (a, b)
                        best_fit = (slope_i, intercept_i, r2_i)

            if best is None or best_fit is None:
                raise RuntimeError("No valid fallback fit window found")
            fit_window, slope, intercept, r2 = best, best_fit[0], best_fit[1], best_fit[2]
    else:
        a, b = window
        t = np.arange(a, b + 1, dtype=np.float64)
        y = np.log(np.clip(survivors_arr[a : b + 1].astype(np.float64), 1.0, None))
        slope, intercept, r2 = _fit_line(t, y)
        fit_window = (a, b)
        cand = 1
        fallback_used = False
        policy_name = "user_provided_window"

    rho = float(-slope)
    survival_curve = (survivors_arr / max(survivors_arr[0], 1)).astype(np.float64)
    return {
        "rho": rho,
        "r2": float(r2),
        "fit_window": [int(fit_window[0]), int(fit_window[1])],
        "survival_curve": survival_curve.tolist(),
        "survivors": survivors_arr.tolist(),
        "slope": float(slope),
        "intercept": float(intercept),
        "window_selection": {
            "policy": policy_name,
            "params": {
                "min_survivors": int(min_survivors),
                "r2_threshold": float(r2_threshold),
                "min_window_len": int(min_window_len),
                "fallback_policy": fallback_policy,
            },
            "chosen_window": [int(fit_window[0]), int(fit_window[1])],
            "r2": float(r2),
            "fallback_used": bool(fallback_used),
            "strict_policy_found_window": bool(not fallback_used),
            "candidate_windows_evaluated": int(cand),
        },
        "fit_window_details": {
            "start": int(fit_window[0]),
            "end": int(fit_window[1]),
            "length": int(fit_window[1] - fit_window[0] + 1),
        },
        "candidate_windows_evaluated": int(cand),
    }
