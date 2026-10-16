"""Reproduce and isolate the original VODE/BDF first-interval failure.

Run: python examples/diagnose_legacy_vode.py
Optional: python examples/diagnose_legacy_vode.py --extra-vode
This file never modifies the simulation library or physical model.
"""
import argparse
from pathlib import Path
import sys
import time
import warnings

import numpy as np
from scipy import integrate, interpolate
from scipy.sparse import csc_matrix

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from modWorm import predefined_classes_nv as nv_classes, utils


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extra-vode", action="store_true", help="Also test VODE nsteps=5000 (may take considerable time)")
    parser.add_argument(
        "--cook-file",
        help="Path to the Cook workbook (or set MODWORM_COOK_CONNECTOME)",
    )
    args = parser.parse_args()
    data = ROOT / "modWorm"
    gap, syn = utils.construct_connectome_Cook(args.cook_file)
    stimulus = np.load(data / "presets_input/input_mat_gentle_post_touch.npy")[:2].astype(float)
    np.random.seed(20261015)
    model = nv_classes.CelegansWorm_NervousSystem_PPC(gap, syn)
    h = float(model.timescale)
    model.input_FDB = np.zeros(model.network_Size, dtype=float)
    model.interp_i_Ext = interpolate.interp1d([0.0, h], stimulus, axis=0, kind="linear", bounds_error=True)
    initial = np.asarray(model.initcond, dtype=float).copy()
    print("state", initial.shape, "clock", h, "input", stimulus.shape)
    print("initial voltage range", float(initial[:model.network_Size].min()), float(initial[:model.network_Size].max()))
    try:
        threshold = model.compute_Vth(stimulus[0])
        print("initial threshold range", float(threshold.min()), float(threshold.max()))
    except Exception as exc:
        print("threshold calculation FAILED:", repr(exc))
        raise

    # BDF method and sparse Jacobian already exist in modWorm/shared_world.py.
    t0 = time.perf_counter()
    try:
        sol = integrate.solve_ivp(
            model.forward_Network, (0., h), initial,
            method="BDF", t_eval=[h],
            jac=lambda t, y: csc_matrix(model.forward_network_Jacobian(t, y)),
            rtol=5e-6, atol=1e-3,
        )
        print("solve_ivp BDF:", "SUCCESS" if sol.success else "FAILED",
              "t_end", float(sol.t[-1]) if len(sol.t) else None,
              "nfev", sol.nfev, "njev", sol.njev,
              "seconds", round(time.perf_counter()-t0, 3), "message", sol.message)
    except Exception as exc:
        print("solve_ivp BDF EXCEPTION", repr(exc))

    for limit in ([500, 5000] if args.extra_vode else [500]):
        t0 = time.perf_counter()
        solver = integrate.ode(model.forward_Network, model.forward_network_Jacobian)
        solver.set_integrator("vode", method="bdf", rtol=5e-6, atol=1e-3, nsteps=limit)
        solver.set_initial_value(initial.copy(), 0.0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                solver.integrate(h)
                print(f"VODE nsteps={limit}:", "SUCCESS" if solver.successful() and abs(solver.t-h)<1e-9 else "FAILED",
                      "reached", float(solver.t), "seconds", round(time.perf_counter()-t0, 3))
            except Exception as exc:
                print(f"VODE nsteps={limit} EXCEPTION", repr(exc))
        for warning in caught:
            print("  warning:", str(warning.message))
    print("NOTE: BDF/VODE success is a smoke check, not a convergence or biological validation.")


if __name__ == "__main__":
    main()
