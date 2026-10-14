"""Compare baseline, zero-dose, and 5-uM serotonin in the Python solver.

Usage:
    python examples/test_serotonin.py CONNECTOME.xls MUSCLE_MAP.xls INPUT.npy

INPUT.npy must be the desired time-by-279 external-current matrix in pA.
Measured behavior is reported, not asserted to have a predetermined direction.
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.special import expit

from modWorm import (
    body_simulations,
    network_params,
    network_simulations,
    predefined_classes_mb,
    predefined_classes_nv,
    utils,
)
from modWorm.serotonin import NEURON_INDICES, SerotoninModulation


AIY_INDICES = np.array([NEURON_INDICES["AIYL"], NEURON_INDICES["AIYR"]])


def run_pair(
    connectome_file: Path,
    muscle_map_file: Path,
    input_file: Path,
    connectome_dataset: str,
    max_steps: int | None = None,
    seed: int = 20261014,
) -> None:
    connectome_loader = {
        "varshney": utils.construct_connectome_Varshney,
        "cook": utils.construct_connectome_Cook,
    }[connectome_dataset]
    gap_connectome, syn_connectome = connectome_loader(str(connectome_file))
    muscle_map = utils.construct_muscle_map_Hall(str(muscle_map_file))
    input_matrix = np.load(input_file)
    if input_matrix.ndim != 2 or input_matrix.shape[1] != 279:
        raise ValueError("INPUT.npy must have shape (timepoints, 279)")
    if max_steps is not None:
        if max_steps < 2:
            raise ValueError("max_steps must be at least 2")
        input_matrix = input_matrix[:max_steps]
    if len(input_matrix) < 2:
        raise ValueError("INPUT.npy must contain at least two timepoints")

    np.random.seed(seed)
    base_network = predefined_classes_nv.CelegansWorm_NervousSystem(
        gap_connectome, syn_connectome
    )
    initial_network_state = base_network.initcond.copy()
    original_gap = base_network.network_Electrical.copy()
    original_syn = base_network.network_Chemical.copy()
    base_body = predefined_classes_mb.CelegansWorm_MuscleBody(muscle_map)

    zero_network = predefined_classes_nv.CelegansWorm_NervousSystem(
        gap_connectome, syn_connectome
    )
    zero_network.initcond = initial_network_state.copy()
    zero_network.serotonin_modulation = SerotoninModulation(concentration=0.0)
    zero_body = predefined_classes_mb.CelegansWorm_MuscleBody(muscle_map)

    serotonin_network = predefined_classes_nv.CelegansWorm_NervousSystem(
        gap_connectome, syn_connectome
    )
    serotonin_network.initcond = initial_network_state.copy()
    serotonin_network.serotonin_modulation = SerotoninModulation(
        concentration=5.0
    )
    serotonin_body = predefined_classes_mb.CelegansWorm_MuscleBody(muscle_map)

    base_solution = network_simulations.run_network(base_network, input_matrix)
    zero_solution = network_simulations.run_network(zero_network, input_matrix)
    serotonin_solution = network_simulations.run_network(
        serotonin_network, input_matrix
    )

    np.testing.assert_array_equal(
        zero_solution["v_solution"], base_solution["v_solution"]
    )
    np.testing.assert_array_equal(
        zero_solution["s_solution"], base_solution["s_solution"]
    )
    np.testing.assert_array_equal(
        zero_solution["v_threshold"], base_solution["v_threshold"]
    )
    for network in (base_network, zero_network, serotonin_network):
        np.testing.assert_array_equal(network.network_Electrical, original_gap)
        np.testing.assert_array_equal(network.network_Chemical, original_syn)

    base_body_solution = body_simulations.run_body(
        base_body, base_network, base_solution
    )
    zero_body_solution = body_simulations.run_body(
        zero_body, zero_network, zero_solution
    )
    serotonin_body_solution = body_simulations.run_body(
        serotonin_body, serotonin_network, serotonin_solution
    )

    np.testing.assert_array_equal(
        zero_body_solution["raw_x_solution"],
        base_body_solution["raw_x_solution"],
    )
    np.testing.assert_array_equal(
        zero_body_solution["raw_y_solution"],
        base_body_solution["raw_y_solution"],
    )

    threshold = serotonin_solution["v_threshold"]
    widths = network_params.CE.B[AIY_INDICES]
    base_activity = expit(
        widths
        * (
            base_solution["v_solution"][:, AIY_INDICES]
            - base_solution["v_threshold"][:, AIY_INDICES]
        )
    )
    serotonin_activity = expit(
        widths
        * (
            serotonin_solution["v_solution"][:, AIY_INDICES]
            - threshold[:, AIY_INDICES]
        )
    )
    ai_y_base = float(np.mean(base_activity))
    ai_y_serotonin = float(np.mean(serotonin_activity))

    def mean_body_speed(body_solution: dict[str, np.ndarray], dt: float) -> float:
        x = body_solution["raw_x_solution"]
        y = body_solution["raw_y_solution"]
        center_x = np.mean(x, axis=1)
        center_y = np.mean(y, axis=1)
        return float(
            np.mean(np.hypot(np.diff(center_x), np.diff(center_y)) / dt)
        )

    dt = serotonin_network.timescale
    base_speed = mean_body_speed(base_body_solution, dt)
    serotonin_speed = mean_body_speed(serotonin_body_solution, dt)

    print(f"AIY mean activation, 0 uM: {ai_y_base:.6g}")
    print(f"AIY mean activation, 5 uM: {ai_y_serotonin:.6g}")
    print(f"Mean body speed, 0 uM: {base_speed:.6g}")
    print(f"Mean body speed, 5 uM: {serotonin_speed:.6g}")
    if max_steps is not None:
        print(
            f"Short-run diagnostic only: {max_steps} samples "
            f"({(max_steps - 1) * serotonin_network.timescale:.3g} s); "
            "do not interpret as a persistent behavioral state."
        )
    print("Zero-serotonin output and connectomes exactly match the base model.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("connectome_file", type=Path)
    parser.add_argument("muscle_map_file", type=Path)
    parser.add_argument("input_file", type=Path)
    parser.add_argument(
        "--connectome-dataset",
        choices=("varshney", "cook"),
        default="varshney",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="optional short diagnostic run; full input is used when omitted",
    )
    parser.add_argument("--seed", type=int, default=20261014)
    args = parser.parse_args()
    run_pair(
        args.connectome_file,
        args.muscle_map_file,
        args.input_file,
        args.connectome_dataset,
        args.max_steps,
        args.seed,
    )


if __name__ == "__main__":
    main()
