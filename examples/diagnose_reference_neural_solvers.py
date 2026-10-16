"""Non-invasive diagnostic: analytical Jacobian, SciPy BDF, optional legacy VODE.

Run from repository root after installing dependencies:
  python examples/diagnose_reference_neural_solvers.py --duration 0.01
  python examples/diagnose_reference_neural_solvers.py --duration 0.01 --legacy

The legacy test is opt-in because it may use hundreds of internal steps and
has previously failed with 'Excess work'. No tolerances or equations are
silently modified, and no path is declared scientifically validated on success.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import csc_matrix

from modWorm.reference_integration import ReferenceConfig, make_full_model_agents
from modWorm.proprioception_simulation import configure_ode_solver


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=0.01)
    parser.add_argument("--legacy", action="store_true", help="Also test legacy SciPy VODE with original settings")
    args = parser.parse_args()
    config = ReferenceConfig(duration=args.duration, steps=1)
    agent = make_full_model_agents(config)[0]
    network, body = agent.nervous_system, agent.muscle_body
    network.interp_i_Ext = lambda t: np.zeros(network.network_Size)
    network.input_FDB = np.zeros(network.network_Size)
    y0 = network.initcond.copy()
    direction = np.random.default_rng(4382).normal(size=len(y0))
    direction /= np.linalg.norm(direction)
    h = 1e-5
    t0 = 0.0
    try:
        j = np.asarray(network.forward_network_Jacobian(t0, y0), dtype=float)
        finite_difference = (
            network.forward_Network(t0, y0 + h * direction)
            - network.forward_Network(t0, y0 - h * direction)
        ) / (2 * h)
        analytical_direction = j @ direction
        jacobian_error = float(np.linalg.norm(finite_difference - analytical_direction) /
                               max(1.0, np.linalg.norm(finite_difference)))
    except Exception as exc:
        print(json.dumps({"stage": "jacobian_audit", "error": str(exc)}, indent=2))
        raise

    start = perf_counter()
    result = solve_ivp(
        network.forward_Network, (0.0, config.duration), y0,
        method="BDF", jac=lambda t, y: csc_matrix(network.forward_network_Jacobian(t, y)),
        rtol=config.neural_rtol, atol=config.neural_atol,
        t_eval=[config.duration],
    )
    summary = {
        "scenario": "unmodulated single-worm neural ODE (no body coupling)",
        "expected_t": config.duration,
        "jacobian_direction_relative_error": jacobian_error,
        "BDF": {
            "success": bool(result.success), "message": str(result.message),
            "reached_t": float(result.t[-1]) if result.t.size else 0.0,
            "wall_seconds": perf_counter() - start,
            "nfev": int(result.nfev), "njev": int(result.njev),
            "nlu": int(result.nlu),
            "all_finite": bool(np.isfinite(result.y).all()),
        },
    }
    if args.legacy:
        neural_solver, _ = configure_ode_solver(
            network, body, False,
            config.neural_rtol, config.neural_atol, config.body_rtol, config.body_atol,
        )
        start = perf_counter()
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            neural_solver.integrate(config.duration)
        success = bool(neural_solver.successful() and abs(neural_solver.t - config.duration) < 1e-10)
        summary["legacy_VODE"] = {
            "success": success, "reached_t": float(neural_solver.t),
            "wall_seconds": perf_counter() - start,
            "warnings": [str(w.message) for w in captured],
            "all_finite": bool(np.isfinite(neural_solver.y).all()),
        }
        if success and result.success:
            summary["legacy_vs_BDF_voltage_max_difference"] = float(np.max(
                np.abs(neural_solver.y - result.y[:, -1])
            ))
    print(json.dumps(summary, indent=2, allow_nan=False))
    if not result.success or not np.isfinite(result.y).all():
        raise SystemExit("SciPy BDF did not produce a complete finite result")


if __name__ == "__main__":
    main()
