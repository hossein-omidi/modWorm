"""Synchronized Python simulation of independent worms in a shared 2-D world."""

from __future__ import annotations

from dataclasses import dataclass
from inspect import signature
from typing import Any

import numpy as np
from scipy import integrate, interpolate
from scipy.sparse import csc_matrix

from modWorm import body_dynamics, body_simulations
from modWorm.collective import capsule_contact_forces
from modWorm.numerics import NumericalIntegrationError, validate_ivp_solution
from modWorm.serotonin.serotonin_modulation import NEURON_INDICES


@dataclass
class WormAgent:
    """One independently configured nervous-system/body pair in a shared world."""

    nervous_system: Any
    muscle_body: Any
    external_current: np.ndarray
    name: str = "worm"


_CONTACT_SENSOR_SITES = {
    "ALML": 0.12,
    "ALMR": 0.12,
    "AVM": 0.28,
    "PVM": 0.72,
    "PLML": 0.88,
    "PLMR": 0.88,
}


def contact_sensory_current(
    nodal_forces: np.ndarray,
    network_size: int,
    *,
    gain_pA_per_force: float,
) -> np.ndarray:
    """Map local contact load to touch-cell current using an explicit proxy.

    Sensor placement fractions are approximate anterior/midbody/posterior
    locations, and the configurable current gain is not biologically fitted.
    """
    forces = np.asarray(nodal_forces, dtype=float)
    if forces.ndim != 2 or forces.shape[1] != 2 or forces.shape[0] < 2:
        raise ValueError("nodal_forces must have shape (at least two points, 2)")
    if not np.isfinite(forces).all():
        raise ValueError("nodal_forces must be finite")
    if not np.isfinite(gain_pA_per_force) or gain_pA_per_force < 0.0:
        raise ValueError("gain_pA_per_force must be finite and nonnegative")

    current = np.zeros(network_size, dtype=float)
    magnitudes = np.linalg.norm(forces, axis=1)
    for neuron_name, fraction in _CONTACT_SENSOR_SITES.items():
        neuron_index = NEURON_INDICES[neuron_name]
        if neuron_index >= network_size:
            raise ValueError(f"network does not contain contact sensor {neuron_name}")
        point_index = int(round(fraction * (len(forces) - 1)))
        current[neuron_index] += gain_pA_per_force * magnitudes[point_index]
    return current


def _body_centerline_and_velocity(
    body: Any, body_state: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    count = body.body_SegCount
    q, qdot = np.split(body_state, 2)
    x1, y1, phi = q[0], q[1], q[2 : 2 + count]
    xdot1, ydot1, phidot = qdot[0], qdot[1], qdot[2 : 2 + count]
    x, y, xdot, ydot = body_dynamics.fwd_body_Coordinates(
        body, x1, y1, xdot1, ydot1, phi, phidot
    )
    centerline = np.column_stack((x, y))
    velocity = np.column_stack((xdot, ydot))
    return centerline, velocity


def _contact_forces(
    agents: list[WormAgent],
    body_states: list[np.ndarray],
    *,
    stiffness: float,
    damping: float,
    radii: np.ndarray,
) -> list[np.ndarray]:
    forces = [
        np.zeros((agent.muscle_body.body_SegCount, 2), dtype=float)
        for agent in agents
    ]
    centerlines = [
        _body_centerline_and_velocity(agent.muscle_body, state)
        for agent, state in zip(agents, body_states)
    ]
    bounds = [
        (
            centerline.min(axis=0) - radius,
            centerline.max(axis=0) + radius,
        )
        for (centerline, _), radius in zip(centerlines, radii)
    ]
    for index_a in range(len(agents)):
        for index_b in range(index_a + 1, len(agents)):
            lower_a, upper_a = bounds[index_a]
            lower_b, upper_b = bounds[index_b]
            if np.any(upper_a < lower_b) or np.any(upper_b < lower_a):
                continue
            centerline_a, velocity_a = centerlines[index_a]
            centerline_b, velocity_b = centerlines[index_b]
            pair_force_a, pair_force_b = capsule_contact_forces(
                centerline_a,
                centerline_b,
                radius_a=float(radii[index_a]),
                radius_b=float(radii[index_b]),
                stiffness=stiffness,
                damping=damping,
                velocity_a=velocity_a,
                velocity_b=velocity_b,
            )
            forces[index_a] += pair_force_a
            forces[index_b] += pair_force_b
    return forces


def _validate_agent(agent: WormAgent, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    network = agent.nervous_system
    body = agent.muscle_body
    if network is body:
        raise ValueError("each worm must have distinct neural and body model instances")
    if not hasattr(network, "forward_Network") or not hasattr(body, "forward_Body"):
        raise TypeError("each worm must provide modWorm neural and body dynamics")

    current = np.asarray(agent.external_current, dtype=float)
    network_size = int(network.network_Size)
    if current.ndim == 1:
        if current.shape != (network_size,):
            raise ValueError("a constant external_current must match network_Size")
        current = np.broadcast_to(current, (len(times), network_size)).copy()
    if current.shape != (len(times), network_size) or not np.isfinite(current).all():
        raise ValueError(
            "external_current must be finite with shape (time samples, network_Size)"
        )

    neural_state = np.asarray(network.initcond, dtype=float)
    body_state = np.asarray(body.initcond, dtype=float)
    if neural_state.ndim != 1 or neural_state.size < 2 * network_size:
        raise ValueError("nervous-system initial state must contain voltage and synaptic states")
    if body_state.shape != (2 * (2 + body.body_SegCount),):
        raise ValueError("body initial state has an unexpected shape")
    if not np.isfinite(neural_state).all() or not np.isfinite(body_state).all():
        raise ValueError("initial neural and body states must be finite")
    return current, neural_state.copy()


def run_shared_world(
    agents: list[WormAgent] | tuple[WormAgent, ...],
    times: np.ndarray,
    *,
    mechanical_contact: bool = True,
    sensory_contact: bool = False,
    contact_stiffness: float = 10.0,
    contact_damping: float = 0.0,
    contact_radii: float | tuple[float, ...] | list[float] | np.ndarray | None = None,
    sensory_gain_pA_per_force: float = 0.0,
    neural_rtol: float = 5e-6,
    neural_atol: float = 1e-3,
    body_rtol: float = 2.5e-6,
    body_atol: float = 1e-3,
) -> dict[str, Any]:
    """Advance all worms on one clock, coupling their body ODEs during contact.

    Neural integrations and first-order PDFR-1 state updates use each shared
    time interval. The body states are then integrated together over that
    interval; capsule forces are recomputed at every body-solver evaluation.
    Mechanosensory current is sampled at the shared interval boundary and
    remains an independently switchable neural input.
    """
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or len(times) < 2:
        raise ValueError("times must contain at least two samples")
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0.0):
        raise ValueError("times must be finite and strictly increasing")
    if not agents:
        raise ValueError("the shared world requires at least one worm")
    if len({id(agent.nervous_system) for agent in agents}) != len(agents):
        raise ValueError("worms must not share nervous-system instances")
    if len({id(agent.muscle_body) for agent in agents}) != len(agents):
        raise ValueError("worms must not share body instances")
    if not np.isfinite(contact_stiffness) or contact_stiffness < 0.0:
        raise ValueError("contact_stiffness must be finite and nonnegative")
    if not np.isfinite(contact_damping) or contact_damping < 0.0:
        raise ValueError("contact_damping must be finite and nonnegative")
    if not np.isfinite(sensory_gain_pA_per_force) or sensory_gain_pA_per_force < 0.0:
        raise ValueError("sensory_gain_pA_per_force must be finite and nonnegative")
    if contact_radii is None:
        radii = np.asarray(
            [float(np.mean(agent.muscle_body.body_RadiusH)) for agent in agents]
        )
    else:
        radii = np.asarray(contact_radii, dtype=float)
        if radii.ndim == 0:
            radii = np.full(len(agents), float(radii))
        if (
            radii.shape != (len(agents),)
            or not np.isfinite(radii).all()
            or np.any(radii < 0.0)
        ):
            raise ValueError(
                "contact_radii must be a nonnegative scalar or one radius per worm"
            )

    inputs_and_states = [_validate_agent(agent, times) for agent in agents]
    external_inputs = [item[0] for item in inputs_and_states]
    neural_states = [item[1] for item in inputs_and_states]
    for left in range(len(agents)):
        for right in range(left + 1, len(agents)):
            if np.shares_memory(
                agents[left].nervous_system.initcond,
                agents[right].nervous_system.initcond,
            ) or np.shares_memory(
                agents[left].muscle_body.initcond,
                agents[right].muscle_body.initcond,
            ):
                raise ValueError("worms must not share neural or body initial-state arrays")
            left_pdf = getattr(agents[left].nervous_system, "pdfr1_modulation", None)
            right_pdf = getattr(agents[right].nervous_system, "pdfr1_modulation", None)
            if (
                left_pdf is not None
                and right_pdf is not None
                and np.shares_memory(left_pdf.signal_state, right_pdf.signal_state)
            ):
                raise ValueError("worms must not share PDFR-1 signaling state")
    body_states = [np.asarray(agent.muscle_body.initcond, dtype=float).copy() for agent in agents]
    neural_histories = [
        np.full((len(times), agent.nervous_system.network_Size), np.nan)
        for agent in agents
    ]
    synaptic_histories = [np.full_like(history, np.nan) for history in neural_histories]
    threshold_histories = [np.full_like(history, np.nan) for history in neural_histories]
    muscle_histories = [
        np.full((len(times), agent.muscle_body.body_SegCount * 2), np.nan)
        for agent in agents
    ]
    body_histories = [np.full((len(times), len(state)), np.nan) for state in body_states]
    contact_histories = [
        np.zeros((len(times), agent.muscle_body.body_SegCount, 2), dtype=float)
        for agent in agents
    ]
    sensory_histories = [
        np.zeros((len(times), agent.nervous_system.network_Size), dtype=float)
        for agent in agents
    ]
    pdf_histories: list[np.ndarray | None] = []

    for index, agent in enumerate(agents):
        network = agent.nervous_system
        network.interp_i_Ext = interpolate.interp1d(
            times, external_inputs[index], axis=0, kind="linear", bounds_error=True
        )
        network.input_FDB = np.zeros(network.network_Size, dtype=float)
        pdf = getattr(network, "pdfr1_modulation", None)
        if pdf is None:
            pdf_histories.append(None)
        else:
            pdf_histories.append(
                np.full((len(times), len(pdf.target_indices)), np.nan)
            )
            pdf_histories[index][0] = pdf.signal_state
        neural_histories[index][0] = neural_states[index][: network.network_Size]
        synaptic_histories[index][0] = neural_states[index][
            network.network_Size : 2 * network.network_Size
        ]
        body_histories[index][0] = body_states[index]

    muscle_forces: list[np.ndarray] = [
        np.zeros(agent.muscle_body.body_SegCount * 2, dtype=float)
        for agent in agents
    ]
    neural_nfev = np.zeros(len(agents), dtype=int)
    neural_njev = np.zeros(len(agents), dtype=int)
    body_nfev = 0

    def sample_contact(current_states: list[np.ndarray]) -> list[np.ndarray]:
        return _contact_forces(
            list(agents),
            current_states,
            stiffness=contact_stiffness,
            damping=contact_damping,
            radii=radii,
        )

    for step, (time_start, time_stop) in enumerate(zip(times[:-1], times[1:])):
        dt = float(time_stop - time_start)
        active_forces = (
            sample_contact(body_states)
            if mechanical_contact or sensory_contact
            else [
                np.zeros((agent.muscle_body.body_SegCount, 2), dtype=float)
                for agent in agents
            ]
        )
        contact_inputs = []
        for index, agent in enumerate(agents):
            neural_size = agent.nervous_system.network_Size
            sensory = (
                contact_sensory_current(
                    active_forces[index],
                    neural_size,
                    gain_pA_per_force=sensory_gain_pA_per_force,
                )
                if sensory_contact
                else np.zeros(neural_size, dtype=float)
            )
            contact_inputs.append(sensory)
            sensory_histories[index][step] = sensory
            contact_histories[index][step] = active_forces[index]

        for index, agent in enumerate(agents):
            network = agent.nervous_system
            body = agent.muscle_body
            external = external_inputs[index][step] + contact_inputs[index]
            threshold = network.compute_Vth(external)
            voltage = neural_states[index][: network.network_Size]
            muscle_forces[index] = body.forward_Muscles(voltage, threshold)
            threshold_histories[index][step] = threshold
            muscle_histories[index][step] = muscle_forces[index]

            proprioceptive = getattr(network, "proprioceptive_FDB", None)
            if proprioceptive is not None:
                network.input_FDB = proprioceptive(
                    body,
                    neural_histories[index],
                    synaptic_histories[index],
                    threshold_histories[index],
                    muscle_histories[index],
                    body_histories[index][:, : 2 + body.body_SegCount],
                    step,
                )

            pdf = getattr(network, "pdfr1_modulation", None)
            if pdf is not None:
                pdf_state = pdf.advance(0.5 * dt, float(time_start))
            else:
                pdf_state = None

            def rhs(
                t: float,
                state: np.ndarray,
                model: Any = network,
                stimulus: np.ndarray = contact_inputs[index],
                signal: np.ndarray | None = pdf_state,
            ) -> np.ndarray:
                kwargs: dict[str, np.ndarray] = {
                    "contact_input_current": stimulus
                }
                if signal is not None:
                    kwargs["pdf_signal_state"] = signal
                return model.forward_Network(t, state, **kwargs)
            kwargs: dict[str, Any] = {}
            jacobian = getattr(network, "forward_network_Jacobian", None)
            if callable(jacobian):
                accepts_contact = (
                    "contact_input_current" in signature(jacobian).parameters
                )
                if accepts_contact or not np.any(contact_inputs[index]):
                    kwargs["jac"] = (
                        lambda t, state, model=network, stimulus=contact_inputs[index]:
                        csc_matrix(
                            model.forward_network_Jacobian(
                                t,
                                state,
                                **(
                                    {"contact_input_current": stimulus}
                                    if accepts_contact
                                    else {}
                                ),
                            )
                        )
                    )
            neural_solution = integrate.solve_ivp(
                rhs,
                (float(time_start), float(time_stop)),
                neural_states[index],
                method="BDF",
                t_eval=np.array([time_stop]),
                rtol=neural_rtol,
                atol=neural_atol,
                **kwargs,
            )
            neural_states[index] = validate_ivp_solution(
                neural_solution,
                np.array([time_stop]),
                len(neural_states[index]),
                operation=f"{agent.name} neural integration at sample {step + 1}",
            )[0]
            neural_nfev[index] += neural_solution.nfev
            neural_njev[index] += neural_solution.njev
            if pdf is not None:
                pdf_state = pdf.advance(0.5 * dt, float(time_start + 0.5 * dt))
                pdf_histories[index][step + 1] = pdf_state
            neural_histories[index][step + 1] = neural_states[index][
                : network.network_Size
            ]
            synaptic_histories[index][step + 1] = neural_states[index][
                network.network_Size : 2 * network.network_Size
            ]

        body_sizes = [len(state) for state in body_states]
        body_initial = np.concatenate(body_states)
        muscle_step_forces = [force.copy() for force in muscle_forces]

        def coupled_body_rhs(time: float, state: np.ndarray) -> np.ndarray:
            local_states: list[np.ndarray] = []
            offset = 0
            for size in body_sizes:
                local_states.append(state[offset : offset + size])
                offset += size
            forces = (
                sample_contact(local_states)
                if mechanical_contact
                else [
                    np.zeros((agent.muscle_body.body_SegCount, 2), dtype=float)
                    for agent in agents
                ]
            )
            derivatives = [
                agent.muscle_body.forward_Body(
                    time,
                    local_states[index],
                    muscle_step_forces[index],
                    forces[index] if mechanical_contact else None,
                )
                for index, agent in enumerate(agents)
            ]
            return np.concatenate(derivatives)

        body_solution = integrate.solve_ivp(
            coupled_body_rhs,
            (float(time_start), float(time_stop)),
            body_initial,
            method="RK45",
            t_eval=np.array([time_stop]),
            max_step=dt,
            rtol=body_rtol,
            atol=body_atol,
        )
        body_trajectory = validate_ivp_solution(
            body_solution,
            np.array([time_stop]),
            len(body_initial),
            operation=f"shared body integration at sample {step + 1}",
        )[0]
        body_nfev += body_solution.nfev
        offset = 0
        for index, agent in enumerate(agents):
            size = body_sizes[index]
            body_states[index] = body_trajectory[offset : offset + size]
            offset += size
            body_histories[index][step + 1] = body_states[index]

    final_forces = (
        sample_contact(body_states)
        if mechanical_contact or sensory_contact
        else [
            np.zeros((agent.muscle_body.body_SegCount, 2), dtype=float)
            for agent in agents
        ]
    )
    for index, agent in enumerate(agents):
        contact_histories[index][-1] = final_forces[index]
        sensory_histories[index][-1] = (
            contact_sensory_current(
                final_forces[index],
                agent.nervous_system.network_Size,
                gain_pA_per_force=sensory_gain_pA_per_force,
            )
            if sensory_contact
            else 0.0
        )
        network = agent.nervous_system
        external = external_inputs[index][-1] + sensory_histories[index][-1]
        threshold = network.compute_Vth(external)
        threshold_histories[index][-1] = threshold
        muscle_histories[index][-1] = agent.muscle_body.forward_Muscles(
            neural_states[index][: network.network_Size], threshold
        )
        if not all(
            np.isfinite(values).all()
            for values in (
                neural_histories[index],
                synaptic_histories[index],
                threshold_histories[index],
                muscle_histories[index],
                body_histories[index],
                contact_histories[index],
                sensory_histories[index],
            )
        ):
            raise NumericalIntegrationError(
                f"{agent.name} simulation left non-finite or uncomputed output samples"
            )

    results = []
    for index, agent in enumerate(agents):
        body = agent.muscle_body
        state_history = body_histories[index]
        q_history = state_history[:, : 2 + body.body_SegCount]
        phi = q_history[:, 2 : 2 + body.body_SegCount]
        raw_x, raw_y = body_simulations.solve_xy(
            body, q_history[:, 0], q_history[:, 1], phi
        )
        smooth_x, smooth_y = body_simulations.postprocess_xy(body, raw_x, raw_y)
        result = {
            "name": agent.name,
            "time": times.copy(),
            "v_solution": neural_histories[index],
            "s_solution": synaptic_histories[index],
            "v_threshold": threshold_histories[index],
            "muscle_force": muscle_histories[index],
            "body_state": state_history,
            "raw_x_solution": raw_x,
            "raw_y_solution": raw_y,
            "x_solution": smooth_x,
            "y_solution": smooth_y,
            "contact_force": contact_histories[index],
            "contact_sensory_current": sensory_histories[index],
        }
        if pdf_histories[index] is not None:
            result["pdf_signal_solution"] = pdf_histories[index]
        results.append(result)

    return {
        "time": times.copy(),
        "agents": results,
        "mechanical_contact": bool(mechanical_contact),
        "sensory_contact": bool(sensory_contact),
        "solver": {
            "neural": "BDF",
            "body": "RK45",
            "neural_nfev": neural_nfev,
            "neural_njev": neural_njev,
            "body_nfev": body_nfev,
        },
    }


def run_two_worms(
    worm_a: WormAgent,
    worm_b: WormAgent,
    times: np.ndarray,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run two distinct worms with optional, independently configured couplings."""
    return run_shared_world([worm_a, worm_b], times, **kwargs)


def run_single_worm(
    worm: WormAgent, times: np.ndarray, **kwargs: Any
) -> dict[str, Any]:
    """Run one worm with the same neural/muscle/body integration used by a world."""
    return run_shared_world([worm], times, **kwargs)
