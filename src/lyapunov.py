"""Conditioned Lyapunov exponent computation with splitting/cloning.

Doctrine reference: constants.DOCTRINE governs time unit and escape convention.
"""

from __future__ import annotations

import numpy as np

from src.system import DelayedOpenExpandingSystem


def lyapunov_conditioned_split(
    system: DelayedOpenExpandingSystem,
    Z0: np.ndarray,
    T: int,
    burn: int,
    seed: int,
    M_target: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    states = np.asarray(Z0, dtype=np.float64).copy()
    if states.ndim != 2 or states.shape[1] != 2:
        raise ValueError("Z0 must be shape (N,2)")
    if M_target is None:
        M_target = int(states.shape[0])

    tangents = np.tile(np.array([1.0, 0.0], dtype=np.float64), (states.shape[0], 1))
    log_sum = 0.0
    n_logs = 0
    survivor_counts: list[int] = []

    for t in range(T):
        x = states[:, 0]
        y = states[:, 1]
        x_next, y_next = system.step(x, y)
        escaped = np.asarray(system.escaped_new_x(x_next), dtype=bool)
        survivors = ~escaped
        survivor_idx = np.flatnonzero(survivors)
        survivor_counts.append(int(survivor_idx.size))
        if survivor_idx.size == 0:
            break

        # Jacobian evaluated at (x_t, y_t), before the map step.
        new_tangents = tangents.copy()
        for i in survivor_idx:
            J = system.jacobian(float(x[i]), float(y[i]))
            v = J @ tangents[i]
            norm_v = float(np.linalg.norm(v))
            if t >= burn and norm_v > 0:
                log_sum += np.log(norm_v)
                n_logs += 1
            if norm_v == 0:
                v = np.array([1.0, 0.0], dtype=np.float64)
                norm_v = 1.0
            new_tangents[i] = v / norm_v

        states = np.column_stack((x_next[survivor_idx], y_next[survivor_idx])).astype(np.float64)
        tangents = new_tangents[survivor_idx]

        if states.shape[0] < M_target:
            clone_count = M_target - states.shape[0]
            picks = rng.integers(0, states.shape[0], size=clone_count)
            states = np.vstack((states, states[picks]))
            tangents = np.vstack((tangents, tangents[picks]))

    lyap = float(log_sum / n_logs) if n_logs > 0 else float("nan")
    return {
        "lyap": lyap,
        "n_logs": int(n_logs),
        "mean_survivors": float(np.mean(survivor_counts)) if survivor_counts else 0.0,
        "min_survivors": int(np.min(survivor_counts)) if survivor_counts else 0,
    }
