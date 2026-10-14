"""Validation helpers for complete, finite numerical integration results."""

from __future__ import annotations

from typing import Any

import numpy as np


class NumericalIntegrationError(RuntimeError):
    """An integration did not produce the complete requested trajectory."""


def validate_ivp_solution(
    solution: Any,
    expected_times: np.ndarray,
    state_size: int,
    *,
    operation: str,
) -> np.ndarray:
    """Return a checked time-major trajectory or raise with solver diagnostics."""
    expected_times = np.asarray(expected_times, dtype=float)
    actual_times = np.asarray(solution.t, dtype=float)
    values = np.asarray(solution.y, dtype=float)
    reached = actual_times[-1] if actual_times.size else None

    if not solution.success:
        raise NumericalIntegrationError(
            f"{operation} failed at t={reached}: {solution.message}"
        )
    if (
        expected_times.ndim != 1
        or expected_times.size < 1
        or not np.isfinite(expected_times).all()
        or actual_times.shape != expected_times.shape
        or values.shape != (state_size, expected_times.size)
        or not np.isfinite(actual_times).all()
        or not np.isfinite(values).all()
        or not np.allclose(actual_times, expected_times, rtol=1e-12, atol=1e-12)
    ):
        raise NumericalIntegrationError(
            f"{operation} returned an incomplete, mistimed, or non-finite trajectory "
            f"(expected {expected_times.size} samples through "
            f"t={expected_times[-1] if expected_times.size else 'undefined'}, "
            f"received {actual_times.size} samples through {reached})"
        )

    return values.T


def validate_ode_step(
    solver: Any,
    expected_time: float,
    state_size: int,
    *,
    operation: str,
    step_index: int,
) -> np.ndarray:
    """Validate one legacy ``scipy.integrate.ode`` step."""
    state = np.asarray(solver.y, dtype=float)
    reached = float(solver.t)
    if (
        not solver.successful()
        or not np.isfinite(reached)
        or not np.isclose(reached, expected_time, rtol=1e-10, atol=1e-12)
        or state.shape != (state_size,)
        or not np.isfinite(state).all()
    ):
        raise NumericalIntegrationError(
            f"{operation} failed or stopped early at sample {step_index}: "
            f"expected t={expected_time}, reached t={reached}, "
            f"solver state size={state.size}, expected={state_size}"
        )
    return state