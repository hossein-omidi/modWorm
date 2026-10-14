"""Static MOD-1 modulation driven by an externally supplied serotonin signal."""

from collections.abc import Callable, Sequence
from math import isfinite
from numbers import Real

import numpy as np


NEURON_INDICES = {
    "ASIL": 75,
    "ASIR": 89,
    "AIYR": 133,
    "AIYL": 137,
    "RIFR": 148,
    "RIFL": 158,
}

ConcentrationControl = float | Callable[[float], float]


class SerotoninModulation:
    """MOD-1 chloride current driven by a scalar or time-dependent 5-HT control.

    Current is returned in pA for voltage in mV and conductance in nS. The
    receptor activation method can be overridden by a kinetic receptor model.
    """

    def __init__(
        self,
        g_mod1: float = 2.0,
        k_d: float = 1.0,
        hill_coefficient: float = 1.0,
        e_cl: float = -70.0,
        concentration: ConcentrationControl = 0.0,
        target_neurons: Sequence[str] = ("AIYL", "AIYR"),
        enabled: bool = True,
    ) -> None:
        self.g_mod1 = self._positive_or_zero("g_mod1", g_mod1)
        self.k_d = self._positive("k_d", k_d)
        self.hill_coefficient = self._positive("hill_coefficient", hill_coefficient)
        self.e_cl = self._finite("e_cl", e_cl)
        self.enabled = bool(enabled)
        self.set_concentration(concentration)

        targets = tuple(target_neurons)
        unknown_targets = set(targets).difference(NEURON_INDICES)
        if unknown_targets:
            raise ValueError(f"Unknown target neuron name(s): {sorted(unknown_targets)}")
        if len(set(targets)) != len(targets):
            raise ValueError("target_neurons must not contain duplicates")
        self.target_neurons = targets
        self.target_indices = tuple(NEURON_INDICES[name] for name in targets)

    def set_concentration(self, concentration: ConcentrationControl) -> None:
        """Set [5-HT] in uM as a constant or callable of simulation time."""
        if callable(concentration):
            self._concentration_control = concentration
            return
        if isinstance(concentration, bool) or not isinstance(concentration, Real):
            raise TypeError("concentration must be a real number or a time callable")
        self._concentration_control = self._nonnegative("concentration", concentration)

    def concentration_at(self, time: float) -> float:
        """Evaluate the external serotonin control at simulation time ``time``."""
        control = self._concentration_control
        value = control(time) if callable(control) else control
        return self._nonnegative("concentration", value)

    def receptor_activation(self, time: float) -> float:
        """Return MOD-1 open fraction using the static Hill steady state."""
        if not self.enabled:
            return 0.0
        concentration = self.concentration_at(time)
        if concentration == 0.0:
            return 0.0
        concentration_power = concentration**self.hill_coefficient
        kd_power = self.k_d**self.hill_coefficient
        return concentration_power / (kd_power + concentration_power)

    def current(self, voltage: float, time: float) -> float:
        """Return the outward-sign MOD-1 current in pA for one target neuron."""
        activation = self.receptor_activation(time)
        return self.g_mod1 * activation * (voltage - self.e_cl)

    def current_vector(self, voltage: np.ndarray, time: float) -> np.ndarray:
        """Return MOD-1 currents in network order; non-target entries are zero."""
        if not self.enabled or not self.target_indices or self.g_mod1 == 0.0:
            return np.zeros_like(voltage, dtype=float)
        activation = self.receptor_activation(time)
        if activation == 0.0:
            return np.zeros_like(voltage, dtype=float)
        if max(self.target_indices) >= len(voltage):
            raise ValueError("network voltage vector does not contain every target neuron")
        currents = np.zeros_like(voltage, dtype=float)
        indices = np.asarray(self.target_indices)
        currents[indices] = self.g_mod1 * activation * (voltage[indices] - self.e_cl)
        return currents

    def conductance_vector(self, time: float, network_size: int) -> np.ndarray:
        """Return dI_MOD-1/dV per neuron, for the network Jacobian."""
        conductances = np.zeros(network_size, dtype=float)
        if self.enabled and self.target_indices and self.g_mod1 != 0.0:
            activation = self.receptor_activation(time)
            if activation == 0.0:
                return conductances
            if max(self.target_indices) >= network_size:
                raise ValueError("network does not contain every target neuron")
            conductances[list(self.target_indices)] = (
                self.g_mod1 * activation
            )
        return conductances

    @staticmethod
    def _finite(name: str, value: Real) -> float:
        if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
            raise ValueError(f"{name} must be a finite real number")
        return float(value)

    @classmethod
    def _positive(cls, name: str, value: Real) -> float:
        number = cls._finite(name, value)
        if number <= 0:
            raise ValueError(f"{name} must be greater than zero")
        return number

    @classmethod
    def _positive_or_zero(cls, name: str, value: Real) -> float:
        number = cls._finite(name, value)
        if number < 0:
            raise ValueError(f"{name} must not be negative")
        return number

    @classmethod
    def _nonnegative(cls, name: str, value: Real) -> float:
        return cls._positive_or_zero(name, value)
