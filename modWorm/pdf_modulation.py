"""Reduced PDFR-1/cAMP signaling with an explicit downstream hypothesis.

PDFR-1 is a G-protein-coupled receptor, not an ion channel. This module
therefore reports receptor occupancy and a slow abstract signaling state;
its optional current is a configurable phenomenological excitability bias,
not an established cell-specific PDFR-1 current law.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import exp, isfinite
from numbers import Real

import numpy as np

from modWorm.serotonin.serotonin_modulation import NEURON_INDICES


ConcentrationControl = float | Callable[[float], float]
DEFAULT_TARGETS = ("AIYL", "AIYR", "RIML", "RIMR", "RIAL", "RIAR")


class PDFR1Modulation:
    """External PDF-1 control and first-order abstract receptor signaling.

    Concentration is in micromolar, ``tau_c`` in seconds, and ``signal`` is a
    dimensionless normalized cAMP-like state. ``advance`` uses the exact
    first-order update for the occupancy at the interval midpoint. Parameter
    defaults are exploratory assumptions, not fitted biological estimates.
    """

    def __init__(
        self,
        concentration: ConcentrationControl = 0.0,
        *,
        k_p: float = 1.0,
        hill_coefficient: float = 1.0,
        tau_c: float = 5.0,
        alpha: float = 1.0,
        coupling_gain_pA_per_signal: float = 0.0,
        target_neurons: Sequence[str] = DEFAULT_TARGETS,
        enabled: bool = True,
    ) -> None:
        self.k_p = self._positive("k_p", k_p)
        self.hill_coefficient = self._positive("hill_coefficient", hill_coefficient)
        self.tau_c = self._positive("tau_c", tau_c)
        self.alpha = self._nonnegative("alpha", alpha)
        self.coupling_gain_pA_per_signal = self._finite(
            "coupling_gain_pA_per_signal", coupling_gain_pA_per_signal
        )
        self.enabled = bool(enabled)
        self.target_neurons = tuple(target_neurons)
        unknown_targets = set(self.target_neurons).difference(NEURON_INDICES)
        if unknown_targets:
            raise ValueError(f"Unknown target neuron name(s): {sorted(unknown_targets)}")
        if len(set(self.target_neurons)) != len(self.target_neurons):
            raise ValueError("target_neurons must not contain duplicates")
        self.target_indices = tuple(NEURON_INDICES[name] for name in self.target_neurons)
        self._concentration_control: ConcentrationControl = 0.0
        self.set_concentration(concentration)
        self.signal_state = np.zeros(len(self.target_indices), dtype=float)

    def set_concentration(self, concentration: ConcentrationControl) -> None:
        if callable(concentration):
            self._concentration_control = concentration
        else:
            self._concentration_control = self._nonnegative("concentration", concentration)

    def concentration_at(self, time: float) -> float:
        control = self._concentration_control
        concentration = control(time) if callable(control) else control
        return self._nonnegative("concentration", concentration)

    def receptor_occupancy(self, time: float) -> float:
        if not self.enabled:
            return 0.0
        concentration = self.concentration_at(time)
        if concentration == 0.0:
            return 0.0
        dose = concentration**self.hill_coefficient
        affinity = self.k_p**self.hill_coefficient
        return dose / (affinity + dose)

    def advance(self, dt: float, time: float) -> np.ndarray:
        """Advance cAMP-like signaling one step; return a copy of target states."""
        dt = self._positive("dt", dt)
        occupancy = self.receptor_occupancy(time + 0.5 * dt)
        target = self.alpha * occupancy
        decay = exp(-dt / self.tau_c)
        self.signal_state = target + (self.signal_state - target) * decay
        return self.signal_state.copy()

    def hypothesis_current_vector(
        self,
        network_size: int,
        *,
        gain_pA_per_signal: float | None = None,
    ) -> np.ndarray:
        """Return an optional, non-channel excitability-bias hypothesis.

        A zero default keeps PDFR-1 signaling observational until an explicitly
        selected downstream effect is calibrated. Positive gain is depolarizing
        in the modWorm RHS convention; negative gain is hyperpolarizing.
        """
        gain = (
            self.coupling_gain_pA_per_signal
            if gain_pA_per_signal is None
            else self._finite("gain_pA_per_signal", gain_pA_per_signal)
        )
        current = np.zeros(network_size, dtype=float)
        if not self.enabled or not self.target_indices:
            return current
        if max(self.target_indices) >= network_size:
            raise ValueError("network does not contain every PDFR-1 target neuron")
        current[list(self.target_indices)] = gain * self.signal_state
        return current

    @staticmethod
    def _finite(name: str, value: Real) -> float:
        if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
            raise ValueError(f"{name} must be a finite real number")
        return float(value)

    @classmethod
    def _positive(cls, name: str, value: Real) -> float:
        number = cls._finite(name, value)
        if number <= 0.0:
            raise ValueError(f"{name} must be greater than zero")
        return number

    @classmethod
    def _nonnegative(cls, name: str, value: Real) -> float:
        number = cls._finite(name, value)
        if number < 0.0:
            raise ValueError(f"{name} must not be negative")
        return number
