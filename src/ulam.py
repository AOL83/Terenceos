"""Deterministic Ulam discretization and sparse spectrum diagnostics.

Doctrine reference: constants.ULAM_ORIENTATION_DESCRIPTION defines row orientation.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import ArpackNoConvergence, eigs

from src.system import DelayedOpenExpandingSystem


def build_ulam_matrix(system: DelayedOpenExpandingSystem, M: int, r: int) -> csr_matrix:
    n_states = M * M
    sample_offsets = (np.arange(r, dtype=np.float64) + 0.5) / r
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    for i in range(M):
        for j in range(M):
            row_idx = i * M + j
            x0 = (i + sample_offsets) / M
            y0 = (j + sample_offsets) / M
            xx, yy = np.meshgrid(x0, y0, indexing="ij")
            x = xx.reshape(-1).astype(np.float64)
            y = yy.reshape(-1).astype(np.float64)
            x_next, y_next = system.step(x, y)
            keep = ~np.asarray(system.escaped_new_x(x_next), dtype=bool)
            if np.any(keep):
                xi = np.floor(x_next[keep] * M).astype(int)
                yi = np.floor(y_next[keep] * M).astype(int)
                xi = np.clip(xi, 0, M - 1)
                yi = np.clip(yi, 0, M - 1)
                dest = xi * M + yi
                unique, counts = np.unique(dest, return_counts=True)
                norm = float(r * r)
                rows.extend([row_idx] * len(unique))
                cols.extend(unique.tolist())
                vals.extend((counts.astype(np.float64) / norm).tolist())

    P = csr_matrix((np.array(vals, dtype=np.float64), (rows, cols)), shape=(n_states, n_states), dtype=np.float64)
    return P


def compute_ulam_spectrum(P: csr_matrix, k: int = 4, maxiter: int = 2000, tol: float = 1e-10) -> dict:
    k_eff = min(k, P.shape[0] - 2) if P.shape[0] > 2 else 1
    warnings: list[str] = []
    converged = True
    try:
        eigvals, _ = eigs(P.T, k=k_eff, which="LM", maxiter=maxiter, tol=tol)
    except ArpackNoConvergence as exc:
        eigvals = exc.eigenvalues
        converged = False
        warnings.append("ARPACK did not fully converge; using available eigenvalues")

    eigvals = np.asarray(eigvals)
    if eigvals.size == 0:
        raise RuntimeError("No eigenvalues computed for Ulam matrix")

    order = np.argsort(-np.abs(eigvals))
    eigvals_sorted = eigvals[order]
    lambda1 = complex(eigvals_sorted[0])
    lambda2 = complex(eigvals_sorted[1]) if eigvals_sorted.size > 1 else complex(np.nan, np.nan)

    return {
        "lambda1": float(np.real(lambda1)),
        "lambda2": [float(np.real(lambda2)), float(np.imag(lambda2))],
        "eigvals_real": np.real(eigvals_sorted).astype(np.float64).tolist(),
        "eigvals_imag": np.imag(eigvals_sorted).astype(np.float64).tolist(),
        "indices_chosen": order.astype(int).tolist(),
        "solver": {"k": int(k_eff), "maxiter": int(maxiter), "tol": float(tol)},
        "n_states": int(P.shape[0]),
        "nnz": int(P.nnz),
        "dtype": str(P.dtype),
        "converged": bool(converged),
        "warnings": warnings,
    }
