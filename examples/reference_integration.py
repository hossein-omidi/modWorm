"""Command-line full modWorm reference scenario using the existing shared-world solver.

Run from repository root:
  python examples/reference_integration.py --scenario baseline --duration 0.01
  python examples/reference_integration.py --scenario serotonin --refine

A single smoke run is not evidence of convergence. ``--refine`` compares
H, H/2 and H/4 at common times, at a potentially significant computational cost.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

# Permit direct execution from any working directory while keeping imports explicit.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from modWorm.reference_integration import ReferenceConfig, run_reference, run_refinement


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=[
        "baseline", "serotonin", "pdf", "combined", "independent", "contact", "sensory", "ppc"
    ], default="baseline")
    parser.add_argument("--duration", type=float, default=0.01, help="Physical seconds simulated")
    parser.add_argument("--steps", type=int, default=1, help="Number of macro coupling intervals")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--pdf-gain", type=float, default=0.0,
                        help="Exploratory PDF downstream gain (pA per signal); zero by default")
    parser.add_argument("--sensory-gain", type=float, default=0.0,
                        help="Exploratory contact-to-touch current gain; zero by default")
    parser.add_argument("--contact-stiffness", type=float, default=10.0)
    parser.add_argument("--separation", type=float, default=None,
                        help="Initial x offset between adjacent worms")
    parser.add_argument("--refine", action="store_true", help="H/H2/H4 reference study; excludes PPC")
    parser.add_argument("--out", type=Path, help="Optional JSON summary path, no raw trajectories")
    args = parser.parse_args()

    scenario = args.scenario
    if scenario == "sensory" and args.sensory_gain == 0.0:
        parser.error("sensory scenario requires an explicit, exploratory --sensory-gain; zero disables the pathway")
    config = ReferenceConfig(
        duration=args.duration, steps=args.steps, seed=args.seed,
        worms=2 if scenario in ("independent", "contact", "sensory") else 1,
        serotonin_uM=5.0 if scenario in ("serotonin", "combined") else 0.0,
        pdf_uM=5.0 if scenario in ("pdf", "combined") else 0.0,
        pdf_gain_pA_per_signal=args.pdf_gain,
        sensory_gain_pA_per_force=args.sensory_gain,
        contact_stiffness=args.contact_stiffness,
        ppc=(scenario == "ppc"),
        mechanical_contact=(scenario in ("contact", "sensory")),
        sensory_contact=(scenario == "sensory"),
        # Canonical initial configuration can overlap when intentionally testing contact.
        separation=args.separation if args.separation is not None else (0.5 if scenario in ("contact", "sensory") else 30.0),
    )

    if args.refine:
        study = run_refinement(config)
        summary = {
            "scenario": scenario, "note": study["interpretation"],
            "dt": study["dt"], "elapsed_seconds": study["elapsed_seconds"],
            "coarse_vs_medium": study["coarse_vs_medium"],
            "medium_vs_fine": study["medium_vs_fine"],
        }
    else:
        run = run_reference(config)
        raw = run["result"]
        summary = {
            "scenario": scenario, "simulated_seconds": config.duration,
            "coupling_interval_seconds": config.dt, "worms": config.worms,
            "elapsed_seconds": run["elapsed_seconds"],
            "solver": {
                name: value.tolist() if isinstance(value, np.ndarray) else value
                for name, value in raw["solver"].items()
            },
            "state_shapes": [
                {name: list(np.shape(value)) for name, value in agent.items()
                 if isinstance(value, np.ndarray)}
                for agent in raw["agents"]
            ],
            "warning": "Successful smoke integration is not convergence validation.",
        }

    output = json.dumps(summary, indent=2, allow_nan=False)
    print(output)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
