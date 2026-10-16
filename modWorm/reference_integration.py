"""Reproducible *partitioned* reference experiments using the existing modWorm model.

This module orchestrates :func:`modWorm.shared_world.run_shared_world`; it does
NOT introduce new neural, muscle, or biomechanical equations. Its baseline is
an operator-split numerical approximation, not a monolithic exact reference.

The legacy ``proprioception_simulation.run_network`` remains available but is
not used for these references because the current VODE path can exhaust its
internal step budget. PPC compatibility is restricted to the native discrete
sampling interval until a physical-time delay implementation is validated.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path
from time import perf_counter
from typing import Callable, Mapping, Sequence

import numpy as np

from modWorm import shared_world


_REPO_DATA = Path(__file__).resolve().parent
_OBSERVABLES = (
    "v_solution", "s_solution", "muscle_force", "body_state",
    "pdf_signal_solution", "contact_force", "contact_sensory_current",
)


@dataclass(frozen=True)
class ReferenceConfig:
    """Configuration for one compatible existing shared-world integration.

    ``steps`` means *coupling intervals*, not solver-internal integration steps.
    The integration horizon is ``duration`` and step length is ``duration/steps``.
    ``ppc`` uses a native *discrete index* delay, so only its class's original
    0.01-second time grid is supported in this reference adapter.
    """

    duration: float = 0.01
    steps: int = 1
    worms: int = 1
    seed: int = 11
    ppc: bool = False
    serotonin_uM: float = 0.0
    pdf_uM: float = 0.0
    pdf_gain_pA_per_signal: float = 0.0
    mechanical_contact: bool = False
    sensory_contact: bool = False
    sensory_gain_pA_per_force: float = 0.0
    contact_stiffness: float = 10.0
    contact_damping: float = 0.0
    separation: float = 30.0
    neural_rtol: float = 5e-6
    neural_atol: float = 1e-3
    body_rtol: float = 2.5e-6
    body_atol: float = 1e-3

    def __post_init__(self) -> None:
        if not math.isfinite(self.duration) or self.duration <= 0:
            raise ValueError("duration must be positive and finite")
        if isinstance(self.steps, bool) or not isinstance(self.steps, int) or self.steps < 1:
            raise ValueError("steps must be a positive integer")
        if isinstance(self.worms, bool) or not isinstance(self.worms, int) or self.worms < 1:
            raise ValueError("worms must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        for name in (
            "serotonin_uM", "pdf_uM", "sensory_gain_pA_per_force",
            "contact_stiffness", "contact_damping", "separation",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not math.isfinite(self.pdf_gain_pA_per_signal):
            raise ValueError("pdf_gain_pA_per_signal must be finite")
        for name in ("neural_rtol", "neural_atol", "body_rtol", "body_atol"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")

    @property
    def dt(self) -> float:
        return self.duration / self.steps

    @property
    def times(self) -> np.ndarray:
        return np.linspace(0.0, self.duration, self.steps + 1)

    @property
    def shared_world_options(self) -> dict:
        return {
            "mechanical_contact": self.mechanical_contact,
            "sensory_contact": self.sensory_contact,
            "sensory_gain_pA_per_force": self.sensory_gain_pA_per_force,
            "contact_stiffness": self.contact_stiffness,
            "contact_damping": self.contact_damping,
            "neural_rtol": self.neural_rtol,
            "neural_atol": self.neural_atol,
            "body_rtol": self.body_rtol,
            "body_atol": self.body_atol,
        }


@lru_cache(maxsize=1)
def _canonical_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load canonical Cook connectomes and muscle map once, without fallback data."""
    from modWorm import utils

    gap, syn = utils.construct_connectome_Cook()
    muscles = np.load(_REPO_DATA / "muscle_maps/muscle_map_adjust.npy", allow_pickle=False)
    if gap.shape != (279, 279) or syn.shape != (279, 279) or muscles.shape != (96, 279):
        raise ValueError("canonical Cook connectome/muscle-map dimensions have changed")
    if not all(np.isfinite(a).all() for a in (gap, syn, muscles)):
        raise ValueError("canonical data contain non-finite values")
    for a in (gap, syn, muscles):
        a.flags.writeable = False
    return gap, syn, muscles


def make_full_model_agents(config: ReferenceConfig) -> list[shared_world.WormAgent]:
    """Build distinct original-class worms with reproducible initial neural states.

    Uses the real Cook matrices. Each initial neural perturbation has the
    original model's Gaussian scale (1e-4 * 0.94), but a local seeded RNG.
    No global ``np.random.seed`` mutation or synthetic connectome fallback.
    """
    from modWorm.predefined_classes_nv import (
        CelegansWorm_NervousSystem, CelegansWorm_NervousSystem_PPC,
    )
    from modWorm.predefined_classes_mb import (
        CelegansWorm_MuscleBody, CelegansWorm_MuscleBody_PPC,
    )
    from modWorm.pdf_modulation import PDFR1Modulation
    from modWorm.serotonin.serotonin_modulation import SerotoninModulation

    gap, syn, muscles = _canonical_arrays()
    neuro_class = CelegansWorm_NervousSystem_PPC if config.ppc else CelegansWorm_NervousSystem
    body_class = CelegansWorm_MuscleBody_PPC if config.ppc else CelegansWorm_MuscleBody
    agents = []
    for index in range(config.worms):
        neuro = neuro_class(gap, syn)
        body = body_class(muscles)
        # Preserve the model's distribution, using reproducible per-agent samples.
        rng = np.random.default_rng(np.random.SeedSequence([config.seed, index]))
        neuro.initcond = 1e-4 * rng.normal(0.0, 0.94, len(neuro.initcond))
        if config.ppc and not math.isclose(config.dt, neuro.timescale, rel_tol=0, abs_tol=1e-12):
            raise ValueError(
                "PPC's original feedback history is sample-index based; "
                f"this class requires dt={neuro.timescale:g} s, got {config.dt:g} s. "
                "Use the non-PPC model for coupling-step refinement."
            )
        if config.serotonin_uM:
            neuro.serotonin_modulation = SerotoninModulation(concentration=config.serotonin_uM)
        if config.pdf_uM or config.pdf_gain_pA_per_signal:
            neuro.pdfr1_modulation = PDFR1Modulation(
                concentration=config.pdf_uM,
                coupling_gain_pA_per_signal=config.pdf_gain_pA_per_signal,
            )
        # Spatial translations modify only body initial position, not anatomy.
        body.initcond = np.asarray(body.initcond, dtype=float).copy()
        body.initcond[0] += index * config.separation
        external = np.zeros((config.steps + 1, neuro.network_Size), dtype=float)
        agents.append(shared_world.WormAgent(neuro, body, external, f"worm_{index}"))
    return agents


def run_reference(
    config: ReferenceConfig,
    *,
    agent_factory: Callable[[ReferenceConfig], Sequence[shared_world.WormAgent]] = make_full_model_agents,
) -> dict:
    """Run the existing BDF/RK45 split-world solver; return its native results.

    The reported runtime includes agent construction and integration, to make
    repeated reference runs comparable. A result is not a convergence proof.
    """
    t0 = perf_counter()
    agents = list(agent_factory(config))
    if len(agents) != config.worms:
        raise ValueError("agent_factory returned a different worm count")
    result = shared_world.run_shared_world(agents, config.times, **config.shared_world_options)
    elapsed = perf_counter() - t0
    if len(result["agents"]) != config.worms:
        raise ValueError("shared-world returned an unexpected number of worms")
    return {"config": config, "elapsed_seconds": elapsed, "result": result}


def normalized_difference(a: np.ndarray, b: np.ndarray, *, atol: float, rtol: float) -> dict:
    """Dimensionless max/RMS step-doubling *differences* (not true-error bounds)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("comparison arrays must have the same shape and finite values")
    if not (math.isfinite(atol) and math.isfinite(rtol) and atol > 0 and rtol > 0):
        raise ValueError("atol and rtol must be positive and finite")
    if a.size == 0:
        raise ValueError("comparison arrays must be nonempty")
    scale = atol + rtol * np.maximum(np.abs(a), np.abs(b))
    residual = np.abs(a - b) / scale
    return {"max": float(np.max(residual)), "rms": float(np.sqrt(np.mean(residual**2)))}


def compare_levels(coarse: dict, fine: dict, *, atol: float, rtol: float) -> dict:
    """Compare aligned states at common sample times; never interpolate contacts."""
    old = coarse["result"]
    new = fine["result"]
    tc, tf = np.asarray(old["time"]), np.asarray(new["time"])
    if len(new["agents"]) != len(old["agents"]):
        raise ValueError("cannot compare different worm counts")
    if len(tf) != 2 * (len(tc) - 1) + 1 or not np.allclose(tf[::2], tc, atol=1e-12, rtol=0):
        raise ValueError("reference levels do not align at identical physical times")
    measures: dict[str, dict[str, dict]] = {}
    for old_agent, new_agent in zip(old["agents"], new["agents"]):
        if old_agent["name"] != new_agent["name"]:
            raise ValueError("reference agents differ in ordering or identity")
        name = old_agent["name"]
        metrics = {}
        for key in _OBSERVABLES:
            if key in old_agent or key in new_agent:
                if key not in old_agent or key not in new_agent:
                    raise ValueError(f"mismatched observable {key}")
                metrics[key] = normalized_difference(
                    old_agent[key], np.asarray(new_agent[key])[::2],
                    atol=atol, rtol=rtol,
                )
        measures[name] = metrics
    return measures


def run_refinement(
    config: ReferenceConfig,
    *,
    agent_factory: Callable[[ReferenceConfig], Sequence[shared_world.WormAgent]] = make_full_model_agents,
    atol: float = 1e-3,
    rtol: float = 1e-4,
) -> dict:
    """Execute H, H/2 and H/4 with independent identical initial states.

    Only smooth/non-PPC cases are eligible for this *discrete-operator*
    comparison. Contact cases can be studied, but observed rates are not
    assumed and differences are not guarantees of physical validity.
    """
    if config.ppc:
        raise ValueError("PPC delay uses discrete sample indices; step refinement requires physical-time delay support")
    from dataclasses import replace
    configs = [replace(config, steps=config.steps * multiplier) for multiplier in (1, 2, 4)]
    runs = [run_reference(c, agent_factory=agent_factory) for c in configs]
    coarse_medium = compare_levels(runs[0], runs[1], atol=atol, rtol=rtol)
    medium_fine = compare_levels(runs[1], runs[2], atol=atol, rtol=rtol)
    return {
        "dt": [c.dt for c in configs],
        "elapsed_seconds": [r["elapsed_seconds"] for r in runs],
        "coarse_vs_medium": coarse_medium,
        "medium_vs_fine": medium_fine,
        "runs": runs,
        "interpretation": "Scaled pairwise differences; H/4 is not presumed exact.",
    }
