"""Optional Julia/Sundials execution of the original PPC worm with MOD-1 and PDFR-1.

This module preserves modWorm's existing Julia neural/muscle/body routines.
It supplies only the new receptor terms and calls the original Julia PPC
preparation/integration functions. It does NOT implement Julia fluid PDEs or
mechanically interacting population contact.

The receptor concentration controls are sampled on the simulation output grid
and held right-continuously between points. Under that declared control law,
PDFR-1's first-order signaling solution is exact at all continuous solver times.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np

_REGISTERED = False


@dataclass(frozen=True)
class JuliaRunSettings:
    neural_rtol: float = 5e-6
    neural_atol: float = 1e-3
    body_rtol: float = 2.5e-6
    body_atol: float = 1e-3
    stiff_body: bool = False
    body_maxiters: int = 1_000_000
    verbose_period: float = 0.0

    def __post_init__(self) -> None:
        for name in ("neural_rtol", "neural_atol", "body_rtol", "body_atol"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.body_maxiters < 1:
            raise ValueError("body_maxiters must be positive")
        if not 0.0 <= self.verbose_period <= 1.0:
            raise ValueError("verbose_period must lie in [0, 1]")


def _validate_inputs(network: Any, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    data = np.asarray(inputs, dtype=np.float64)
    n = int(network.network_Size)
    if data.ndim != 2 or data.shape[1] != n or data.shape[0] < 2:
        raise ValueError(f"input_mat must have shape (at least 2, {n})")
    if not np.isfinite(data).all():
        raise ValueError("input_mat must contain only finite values")
    dt = float(network.timescale)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("network.timescale must be finite and positive")
    return np.ascontiguousarray(data), np.arange(len(data), dtype=float) * dt


def _pdf_profile(pdf: Any, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact knot solution for piecewise-constant Hill receptor occupancy.

    State c_j(t+h) = c_inf + (c_j(t)-c_inf)exp(-h/tau_c).
    The same formula evaluates c_j at any solver stage t in the interval.
    """
    length = len(times)
    if pdf is None:
        return np.zeros((length, 0)), np.zeros(length - 1)
    targets = tuple(pdf.target_indices)
    knots = np.empty((length, len(targets)), dtype=float)
    initial = np.asarray(pdf.signal_state, dtype=float)
    if initial.shape != (len(targets),) or not np.isfinite(initial).all():
        raise ValueError("PDF signal_state has an invalid shape or value")
    knots[0] = initial
    equilibrium = np.empty(length - 1, dtype=float)
    for k in range(length - 1):
        # receptor_occupancy(t_k) reads the original module's scalar control.
        occupancy = float(pdf.receptor_occupancy(float(times[k])))
        if not np.isfinite(occupancy) or not 0.0 <= occupancy <= 1.0:
            raise ValueError("invalid PDF receptor occupancy")
        equilibrium[k] = float(pdf.alpha) * occupancy
        delta = float(times[k + 1] - times[k])
        # expm1 retains accuracy for very small dt/tau.
        knots[k + 1] = knots[k] + (equilibrium[k] - knots[k]) * (-np.expm1(-delta / float(pdf.tau_c)))
    return knots, equilibrium


def _configure_modulation(network: Any, times: np.ndarray, serotonin: Any, pdf: Any) -> dict:
    n = int(network.network_Size)
    mod_g = np.zeros(n, dtype=np.float64)
    mod_activation = np.zeros(len(times), dtype=np.float64)
    e_cl = -70.0
    if serotonin is not None and serotonin.enabled:
        for j in serotonin.target_indices:
            if j < 0 or j >= n:
                raise ValueError("MOD-1 target outside network")
            mod_g[j] = float(serotonin.g_mod1)
        e_cl = float(serotonin.e_cl)
        mod_activation[:] = [
            serotonin.receptor_activation(float(t)) for t in times
        ]
        if not np.isfinite(mod_activation).all():
            raise ValueError("serotonin activation must be finite")
    pdf_knots, pdf_eq = _pdf_profile(pdf, times)
    target_ids = np.array(pdf.target_indices if pdf is not None and pdf.enabled else (), dtype=np.int64)
    if np.any(target_ids < 0) or np.any(target_ids >= n):
        raise ValueError("PDF target outside network")
    gain = (float(pdf.coupling_gain_pA_per_signal)
            if pdf is not None and pdf.enabled else 0.0)
    return {
        "modworm_g_mod1": mod_g,
        "modworm_5ht_activation": mod_activation,
        "modworm_e_cl": e_cl,
        "modworm_pdf_knots": pdf_knots,
        "modworm_pdf_equilibria": pdf_eq,
        "modworm_pdf_targets": target_ids + 1,  # Julia's 1-based indexing
        "modworm_pdf_gain": gain,
        "modworm_pdf_tau": float(pdf.tau_c) if pdf is not None else 1.0,
    }



def audit_resting_threshold_factorization(network: Any, *, tolerance: float = 1e-8) -> dict[str, float]:
    """Check that the stored permutation and LU factors solve the threshold system."""
    n = int(network.network_Size)
    g_leak = np.asarray(network.neuron_Linear[0], dtype=float)
    gap = np.asarray(network.network_Electrical, dtype=float)
    chem = np.asarray(network.network_Chemical[0], dtype=float)
    rise = np.asarray(network.neuron_Chemical[0], dtype=float)
    fall = np.asarray(network.neuron_Chemical[1], dtype=float)
    if gap.shape != (n, n) or chem.shape != (n, n):
        raise ValueError("threshold audit requires the original full connectome matrices")
    gap_off = gap.copy()
    np.fill_diagonal(gap_off, 0.0)
    s_eq = np.round(rise / (rise + 2.0 * fall), 4)
    weighted = chem * s_eq[None, :]
    matrix = gap_off - np.diag(g_leak + gap_off.sum(axis=1) + weighted.sum(axis=1))
    permutation = np.asarray(network.P_vth, dtype=float)
    lower = np.asarray(network.LL_vth, dtype=float)
    upper = np.asarray(network.UU_vth, dtype=float)
    if any(factor.shape != (n, n) for factor in (permutation, lower, upper)):
        raise ValueError("threshold audit requires square P, L, and U factors")
    from scipy.linalg import solve_triangular
    rhs = np.asarray(network.bb_vth, dtype=float)
    threshold = solve_triangular(
        upper,
        solve_triangular(
            lower, permutation.T @ rhs, lower=True, check_finite=False
        ),
        check_finite=False
    )
    scale = max(1.0, float(np.linalg.norm(matrix, ord=np.inf)))
    factor_error = float(
        np.linalg.norm(matrix - permutation @ lower @ upper, ord=np.inf) / scale
    )
    rhs_scale = max(1.0, float(np.linalg.norm(rhs, ord=np.inf)))
    residual = float(np.linalg.norm(matrix @ threshold - rhs, ord=np.inf) / rhs_scale)
    if not np.isfinite(factor_error + residual) or max(factor_error, residual) > tolerance:
        raise RuntimeError(
            "Stored resting-threshold P,L,U factors do not solve their intended "
            "linear system. Check network_interactions.init_vth_Linear. "
            f"factor_error={factor_error:.3g}, residual={residual:.3g}"
        )
    return {"factor_relative_error": factor_error, "threshold_residual": residual}

def _register_julia() -> Any:
    global _REGISTERED
    # Lazy import: ordinary Python/test use must not initialize Julia.
    from modWorm import Main, require_julia
    require_julia()
    if not _REGISTERED:
        source = Path(__file__).with_name("_julia_modulated_ppc.jl").read_text(encoding="utf8")
        Main.eval(source)
        _REGISTERED = True
    return Main


def run_modulated_julia(
    neural_model: Any,
    body_model: Any,
    input_mat: np.ndarray,
    *,
    serotonin: Any = None,
    pdfr1: Any = None,
    settings: JuliaRunSettings | None = None,
) -> dict[str, Any]:
    """Evolve one full 279-neuron/24-segment PPC worm using Julia.

    Pass the *Julia* classes: CelegansWorm_NervousSystem_PPC_Julia and
    CelegansWorm_MuscleBody_PPC_Julia. This interface does not convert a Python
    class to a Julia class silently, since their method contracts differ.
    """
    if not neural_model.__class__.__name__.endswith("PPC_Julia"):
        raise TypeError("neural_model must be CelegansWorm_NervousSystem_PPC_Julia")
    if not body_model.__class__.__name__.endswith("PPC_Julia"):
        raise TypeError("body_model must be CelegansWorm_MuscleBody_PPC_Julia")
    if settings is None:
        settings = JuliaRunSettings()
    if not isinstance(settings, JuliaRunSettings):
        raise TypeError("settings must be JuliaRunSettings")
    if serotonin is None:
        serotonin = getattr(neural_model, "serotonin_modulation", None)
    if pdfr1 is None:
        pdfr1 = getattr(neural_model, "pdfr1_modulation", None)
    inputs, times = _validate_inputs(neural_model, input_mat)
    if int(neural_model.network_Size) != len(np.asarray(neural_model.initcond)) // 2:
        raise ValueError("this accelerator expects the base [v, s] neural state")
    expected_body = 2 * (2 + int(body_model.body_SegCount))
    if np.asarray(body_model.initcond).shape != (expected_body,):
        raise ValueError("body initial state does not match segment count")

    threshold_audit = audit_resting_threshold_factorization(neural_model)

    # The existing Julia classes install their original module functions.
    # The new RHS is separately named and never overwrites forward_Network!.
    main = _register_julia()
    neural_model.forward_Network()
    neural_model.compute_Vth()
    neural_model.proprioceptive_FDB()
    body_model.forward_Muscles()
    body_model.forward_Body()

    nn = dict(vars(neural_model))
    # Python receptor instances are not Julia parameter structs. The validated
    # numeric receptor arrays below are the only data the Julia RHS consumes.
    nn.pop("serotonin_modulation", None)
    nn.pop("pdfr1_modulation", None)
    nn.update({
        "input_mat": inputs,
        "timepoints": times,
        "nsteps": len(times),
        "tf": float(times[-1]),
        "input_FDB": np.zeros(neural_model.network_Size, dtype=float),
        "id": 0,
    })
    nn.update(_configure_modulation(neural_model, times, serotonin, pdfr1))

    mb = dict(vars(body_model))
    mb.update({
        "timescale": neural_model.timescale,
        "fR": np.zeros(body_model.body_SegCount - 1),
        "fL": np.zeros(body_model.body_SegCount - 1),
        "stiffness": settings.stiff_body,
        "maxiters": settings.body_maxiters,
        "id": 0,
    })
    t_start = perf_counter()
    output = main.modworm_run_modulated_ppc(
        nn, mb,
        settings.neural_rtol, settings.neural_atol,
        settings.body_rtol, settings.body_atol,
        settings.verbose_period,
    )
    wall_seconds = perf_counter() - t_start

    # Same body coordinate reconstruction as original run_network_julia.
    from modWorm import body_simulations
    x0 = np.asarray(output["x0"], dtype=float)
    y0 = np.asarray(output["y0"], dtype=float)
    phi = np.asarray(output["phi"], dtype=float)
    raw_x, raw_y = body_simulations.solve_xy(body_model, x0, y0, phi)
    x, y = body_simulations.postprocess_xy(body_model, raw_x, raw_y)
    solution = {
        "time": times,
        "v_solution": np.asarray(output["v_solution"], dtype=float),
        "s_solution": np.asarray(output["s_solution"], dtype=float),
        "v_threshold": np.asarray(output["v_threshold"], dtype=float),
        "muscle_force": np.asarray(output["muscle_force"], dtype=float),
        "x_solution": np.asarray(x, dtype=float),
        "y_solution": np.asarray(y, dtype=float),
        "raw_x_solution": np.asarray(raw_x, dtype=float),
        "raw_y_solution": np.asarray(raw_y, dtype=float),
        "phi": phi,
        "solver": "Julia CVODE_BDF + Tsit5/CVODE_BDF; original PPC coupling",
        "elapsed_seconds": wall_seconds,
        "control_law": "right-continuous piecewise constant serotonin/PDF concentration",
        "threshold_audit": threshold_audit,
    }
    if pdfr1 is not None:
        knots = nn["modworm_pdf_knots"]
        solution["pdf_signal_solution"] = knots.copy()
    for key in ("v_solution", "s_solution", "v_threshold", "muscle_force", "phi"):
        if len(solution[key]) != len(times) or not np.isfinite(solution[key]).all():
            raise RuntimeError(f"Julia returned invalid trajectory for {key}")
    if not np.isfinite(solution["x_solution"]).all() or not np.isfinite(solution["y_solution"]).all():
        raise RuntimeError("Julia returned a nonfinite body trajectory")
    # Commit mutable PDF state only after successful integration.
    if pdfr1 is not None:
        pdfr1.signal_state = nn["modworm_pdf_knots"][-1].copy()
    return solution


def run_independent_julia_population(
    population: Sequence[tuple[Any, Any, np.ndarray, Any, Any]],
    *, settings: JuliaRunSettings | None = None
) -> dict[str, Any]:
    """Separate complete Julia worms; no contact and no collective force.

    Each tuple is (neural_model, body_model, inputs, serotonin, pdfr1).
    Observational population statistics do not imply collective behavior.
    """
    if not population:
        raise ValueError("population must contain at least one worm")
    if len({id(row[0]) for row in population}) != len(population):
        raise ValueError("population worms must have distinct neural model instances")
    if len({id(row[1]) for row in population}) != len(population):
        raise ValueError("population worms must have distinct body model instances")
    pdf_objects = [row[4] for row in population if row[4] is not None]
    if len({id(obj) for obj in pdf_objects}) != len(pdf_objects):
        raise ValueError("population worms must not share mutable PDF signaling instances")
    # Reject incompatible sample clocks before launching any costly Julia solve.
    expected_time = None
    for nn, _body, current, _ser, _pdf in population:
        _, clock = _validate_inputs(nn, current)
        if expected_time is not None and not np.array_equal(clock, expected_time):
            raise ValueError("all population worms must have identical sample clocks")
        expected_time = clock
    results = []
    for nn, body, current, ser, pdf in population:
        results.append(run_modulated_julia(
            nn, body, current, serotonin=ser, pdfr1=pdf, settings=settings,
        ))
    times = results[0]["time"]
    if any(not np.array_equal(item["time"], times) for item in results[1:]):
        raise ValueError("all population trajectories must share the same time axis")
    return {"time": times, "agents": results, "mechanical_contact": False,
            "sensory_contact": False, "backend": "independent_julia"}
