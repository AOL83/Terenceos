"""Authoritative delayed open expanding system.

Doctrine reference:
- Time unit: see constants.TIME_UNIT_DESCRIPTION.
- Escape convention: see constants.ESCAPE_CONVENTION_DESCRIPTION.
- Ulam orientation: see constants.ULAM_ORIENTATION_DESCRIPTION.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class SystemParameters:
    theta: float
    mu: float
    kappa: float
    a: float

    def with_updates(self, **updates: float) -> "SystemParameters":
        data = asdict(self)
        data.update(updates)
        return SystemParameters(**data)


class DelayedOpenExpandingSystem:
    def __init__(self, params: SystemParameters):
        self.params = params
        self.validate()

    def validate(self) -> None:
        p = self.params
        if not (0.0 <= p.a < 1.0):
            raise ValueError("a must satisfy 0 <= a < 1")
        if not (0.0 < p.kappa < 0.5):
            raise ValueError("kappa must satisfy 0 < kappa < 0.5")
        if not (0.0 <= p.mu <= 1.0):
            raise ValueError("mu must satisfy 0 <= mu <= 1")
        if not (abs(p.theta) < 0.25):
            raise ValueError("abs(theta) must satisfy < 0.25")

    @staticmethod
    def circle_distance(x: np.ndarray | float, y: float) -> np.ndarray | float:
        diff = np.abs(np.asarray(x) - y)
        return np.minimum(diff, 1.0 - diff)

    def f(self, x: np.ndarray | float) -> np.ndarray | float:
        x_arr = np.asarray(x)
        return np.mod(x_arr * 2.0 + self.params.theta * np.sin(2.0 * np.pi * x_arr), 1.0)

    def f_prime(self, x: np.ndarray | float) -> np.ndarray | float:
        x_arr = np.asarray(x)
        return 2.0 + self.params.theta * 2.0 * np.pi * np.cos(2.0 * np.pi * x_arr)

    def step(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        fx = self.f(x)
        fy = self.f(y)
        x_next = np.mod((1.0 - self.params.mu) * fx + self.params.mu * fy, 1.0)
        y_next = np.asarray(x, dtype=np.float64)
        return np.asarray(x_next, dtype=np.float64), y_next

    def jacobian(self, x: float, y: float) -> np.ndarray:
        return np.array(
            [
                [(1.0 - self.params.mu) * self.f_prime(x), self.params.mu * self.f_prime(y)],
                [1.0, 0.0],
            ],
            dtype=np.float64,
        )

    def escaped_new_x(self, x_next: np.ndarray | float) -> np.ndarray | bool:
        return self.circle_distance(x_next, self.params.a) < self.params.kappa

    def identity_hash(self) -> str:
        def f17g(v: float) -> str:
            return format(float(v), ".17g")

        payload = {
            "a": f17g(self.params.a),
            "kappa": f17g(self.params.kappa),
            "mu": f17g(self.params.mu),
            "theta": f17g(self.params.theta),
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:12]
