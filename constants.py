"""Repository doctrine constants used across the TerenceOS pipeline."""

from __future__ import annotations

import hashlib

TIME_UNIT_DESCRIPTION = (
    "One time unit equals one application of the delayed map step: "
    "(x_t, y_t) -> (x_{t+1}, y_{t+1})."
)

ESCAPE_CONVENTION_DESCRIPTION = (
    "Escape is decided using the newly produced x-coordinate x_{t+1}: a trajectory is "
    "removed at step t when circle_distance(x_{t+1}, a) < kappa."
)

ULAM_ORIENTATION_DESCRIPTION = (
    "Ulam orientation is row-substochastic: rows index source cells and columns index "
    "destination cells; each row is normalized by the number of deterministic samples "
    "started in that source cell, yielding row sums <= 1 in the open system."
)

DOCTRINE = "\n\n".join(
    [
        TIME_UNIT_DESCRIPTION,
        ESCAPE_CONVENTION_DESCRIPTION,
        ULAM_ORIENTATION_DESCRIPTION,
    ]
)

DOCTRINE_HASH = hashlib.sha256(DOCTRINE.encode("utf-8")).hexdigest()[:12]

__all__ = [
    "TIME_UNIT_DESCRIPTION",
    "ESCAPE_CONVENTION_DESCRIPTION",
    "ULAM_ORIENTATION_DESCRIPTION",
    "DOCTRINE",
    "DOCTRINE_HASH",
]
