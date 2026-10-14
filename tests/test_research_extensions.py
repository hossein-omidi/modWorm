import unittest
import warnings
import copy
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy import integrate
from scipy.sparse import csc_matrix

from modWorm.behavior_analysis import locomotion_metrics, population_metrics
from modWorm import body_simulations, network_simulations
from modWorm.body_dynamics import fwd_body_Coordinates, fwd_body_D2Coordinates
from modWorm.collective import capsule_contact_forces
from modWorm.network_interactions import fwd_jacobian12_Linear, fwd_jacobian34_Linear
from modWorm.numerics import NumericalIntegrationError, validate_ivp_solution, validate_ode_step
from modWorm.predefined_classes_mb import CelegansWorm_MuscleBody
from modWorm.predefined_classes_nv import CelegansWorm_NervousSystem
from modWorm.pdf_modulation import PDFR1Modulation
from modWorm.network_dynamics import fwd_activity_Chemical
from modWorm.proprioception_simulation import configure_ode_solver
from modWorm.serotonin.serotonin_modulation import NEURON_INDICES, SerotoninModulation
from modWorm.shared_world import (
    WormAgent,
    _body_centerline_and_velocity,
    _contact_forces,
    contact_sensory_current,
    run_single_worm,
    run_shared_world,
    run_two_worms,
)


class BehavioralMetricTests(unittest.TestCase):
    def test_signed_axial_velocity_and_reversal_are_observational(self):
        time = np.arange(6, dtype=float)
        center_x = np.array([0, 1, 2, 1, 0, -1], dtype=float)
        body_offsets = np.linspace(1.0, -1.0, 9)
        x = center_x[:, None] + body_offsets
        y = np.zeros_like(x)

        measured = locomotion_metrics(time, x, y, speed_threshold=0.1)

        np.testing.assert_allclose(
            measured["signed_axial_velocity"],
            [1.0, 1.0, -1.0, -1.0, -1.0],
        )
        self.assertEqual(measured["forward_bout_count"], 1)
        self.assertEqual(measured["backward_bout_count"], 1)
        self.assertEqual(measured["reversal_count"], 1)
        self.assertAlmostEqual(measured["path_length"], 5.0)
        self.assertAlmostEqual(measured["explored_convex_hull_area"], 0.0)

    def test_population_clustering_and_encounter_duration(self):
        time = np.array([0.0, 1.0, 2.0])
        positions = np.array(
            [
                [[0.0, 0.0], [0.5, 0.0], [5.0, 0.0]],
                [[0.0, 0.0], [2.0, 0.0], [5.0, 0.0]],
                [[0.0, 0.0], [0.5, 0.0], [5.0, 0.0]],
            ]
        )

        measured = population_metrics(time, positions, cluster_distance=1.0, domain_area=10.0)

        np.testing.assert_array_equal(measured["cluster_count"], [1, 0, 1])
        np.testing.assert_array_equal(measured["largest_cluster_size"], [2, 1, 2])
        self.assertAlmostEqual(measured["global_density_per_area"], 0.3)
        self.assertEqual(len(measured["encounters"]), 2)
        self.assertEqual([event["duration"] for event in measured["encounters"]], [1.0, 0.0])

    def test_invalid_time_axis_is_rejected(self):
        with self.assertRaises(ValueError):
            locomotion_metrics(
                np.array([0.0, 0.0]),
                np.zeros((2, 4)),
                np.zeros((2, 4)),
            )


class PDFR1Tests(unittest.TestCase):
    def test_receptor_occupancy_and_exact_first_order_step(self):
        modulation = PDFR1Modulation(concentration=1.0, tau_c=2.0, alpha=0.8)
        self.assertAlmostEqual(modulation.receptor_occupancy(0.0), 0.5)
        state = modulation.advance(2.0, 0.0)
        self.assertAlmostEqual(state[0], 0.4 * (1.0 - np.exp(-1.0)))
        self.assertEqual(modulation.hypothesis_current_vector(279).max(), 0.0)

    def test_pdf_targets_and_loss_of_function(self):
        modulation = PDFR1Modulation(
            concentration=5.0,
            target_neurons=("AIYL", "RIML"),
        )
        modulation.advance(0.1, 0.0)
        current = modulation.hypothesis_current_vector(
            279, gain_pA_per_signal=2.0
        )
        targets = modulation.target_indices
        self.assertGreater(current[targets[0]], 0.0)
        self.assertGreater(current[targets[1]], 0.0)
        self.assertEqual(np.count_nonzero(current), 2)

        knockout = PDFR1Modulation(concentration=5.0, enabled=False)
        self.assertEqual(knockout.receptor_occupancy(0.0), 0.0)
        np.testing.assert_array_equal(knockout.advance(0.1, 0.0), [0.0] * 6)

    def test_pdf_state_is_instance_local(self):
        first = PDFR1Modulation(concentration=5.0)
        second = PDFR1Modulation(concentration=0.0)
        first.advance(1.0, 0.0)
        self.assertTrue(np.all(first.signal_state > 0.0))
        np.testing.assert_array_equal(second.signal_state, np.zeros(6))

    def test_pdf_hypothesis_changes_only_configured_voltage_equations(self):
        size = 279
        network = object.__new__(CelegansWorm_NervousSystem)
        network.network_Size = size
        network.neuron_C = np.ones(size)
        network.neuron_Linear = np.vstack((np.full(size, 0.01), np.full(size, -35.0)))
        network.neuron_Chemical = np.vstack(
            (np.ones(size), np.full(size, 0.3), np.full(size, 0.125))
        )
        network.network_Electrical = np.zeros((size, size))
        network.network_Chemical = np.zeros((2, size, size))
        network.compute_Vth = lambda i_ext: np.zeros(size)
        network.interp_i_Ext = lambda time: np.zeros(size)
        voltage = np.full(size, -35.0)
        state = np.concatenate((voltage, np.zeros(size)))
        baseline = network.forward_Network(0.0, state)
        modulation = PDFR1Modulation(
            concentration=5.0,
            target_neurons=("AIYL", "RIML"),
            coupling_gain_pA_per_signal=2.0,
        )
        modulation.advance(0.1, 0.0)
        network.pdfr1_modulation = modulation

        modulated = network.forward_Network(0.0, state)

        expected = np.zeros(size)
        expected[list(modulation.target_indices)] = (
            modulation.coupling_gain_pA_per_signal * modulation.signal_state
        )
        np.testing.assert_allclose(modulated[:size] - baseline[:size], expected)
        np.testing.assert_array_equal(modulated[size:], baseline[size:])

    def test_pdf_bias_has_zero_voltage_derivative_in_jacobian(self):
        size = 279
        network = SimpleNamespace(
            network_Size=size,
            neuron_C=np.ones(size),
            neuron_Linear=np.zeros((2, size)),
            network_Electrical=np.zeros((size, size)),
            network_Chemical=np.zeros((2, size, size)),
        )
        voltage = np.zeros(size)
        synaptic_state = np.zeros(size)
        base_jacobian, _ = fwd_jacobian12_Linear(network, voltage, synaptic_state)
        modulation = PDFR1Modulation(
            concentration=5.0,
            coupling_gain_pA_per_signal=2.0,
        )
        modulation.advance(0.1, 0.0)
        network.pdfr1_modulation = modulation

        modulated_jacobian, _ = fwd_jacobian12_Linear(
            network, voltage, synaptic_state
        )

        np.testing.assert_array_equal(modulated_jacobian, base_jacobian)

    def test_contact_current_jacobian_matches_rhs_with_neuromodulation(self):
        network = CelegansWorm_NervousSystem(
            np.zeros((279, 279)), np.zeros((279, 279))
        )
        network.interp_i_Ext = lambda time: np.zeros(279)
        network.initcond = np.concatenate(
            (np.full(279, -35.0), np.full(279, 0.1))
        )
        network.serotonin_modulation = SerotoninModulation(concentration=5.0)
        pdf = PDFR1Modulation(
            concentration=5.0, coupling_gain_pA_per_signal=1.0
        )
        pdf.advance(0.1, 0.0)
        network.pdfr1_modulation = pdf
        contact = np.zeros(279)
        contact[NEURON_INDICES["ALML"]] = 2.0
        state = network.initcond.copy()
        time = 0.02
        for stimulus in (np.zeros(279), contact):
            analytic = network.forward_network_Jacobian(
                time, state, contact_input_current=stimulus
            )
            for column in (
                NEURON_INDICES["ALML"],
                network.network_Size + NEURON_INDICES["ALML"],
            ):
                baseline = network.forward_Network(
                    time,
                    state,
                    contact_input_current=stimulus,
                    pdf_signal_state=pdf.signal_state,
                )
                perturbed = state.copy()
                perturbed[column] += 1e-5
                numerical_column = (
                    network.forward_Network(
                        time,
                        perturbed,
                        contact_input_current=stimulus,
                        pdf_signal_state=pdf.signal_state,
                    )
                    - baseline
                ) / 1e-5
                np.testing.assert_allclose(
                    analytic[:, column], numerical_column, rtol=1e-5, atol=1e-6
                )


class ContactMechanicsTests(unittest.TestCase):
    def test_capsule_contact_is_equal_opposite_and_separation_is_force_free(self):
        first = np.array([[0.0, 0.0], [1.0, 0.0]])
        touching = np.array([[0.0, 0.15], [1.0, 0.15]])
        separated = touching + np.array([0.0, 0.1])

        force_a, force_b = capsule_contact_forces(
            first,
            touching,
            radius_a=0.1,
            radius_b=0.1,
            stiffness=10.0,
        )
        self.assertGreater(np.linalg.norm(force_a), 0.0)
        np.testing.assert_allclose(force_a.sum(axis=0) + force_b.sum(axis=0), [0.0, 0.0])

        moment_a = np.sum(first[:, 0] * force_a[:, 1] - first[:, 1] * force_a[:, 0])
        moment_b = np.sum(
            touching[:, 0] * force_b[:, 1] - touching[:, 1] * force_b[:, 0]
        )
        self.assertAlmostEqual(moment_a + moment_b, 0.0, places=12)

        no_force_a, no_force_b = capsule_contact_forces(
            first,
            separated,
            radius_a=0.1,
            radius_b=0.1,
            stiffness=10.0,
        )
        np.testing.assert_array_equal(no_force_a, np.zeros_like(first))
        np.testing.assert_array_equal(no_force_b, np.zeros_like(separated))

    def test_coincident_degenerate_capsules_have_finite_action_reaction(self):
        point = np.zeros((2, 2))
        force_a, force_b = capsule_contact_forces(
            point,
            point,
            radius_a=0.1,
            radius_b=0.1,
            stiffness=10.0,
        )
        self.assertTrue(np.isfinite(force_a).all())
        self.assertTrue(np.isfinite(force_b).all())
        np.testing.assert_allclose(
            force_a.sum(axis=0) + force_b.sum(axis=0), [0.0, 0.0]
        )

    def test_contact_damping_only_opposes_approach(self):
        first = np.array([[0.0, 0.0], [1.0, 0.0]])
        second = np.array([[0.0, 0.15], [1.0, 0.15]])
        velocity_a = np.zeros_like(first)
        velocity_b = np.zeros_like(second)
        velocity_a[:, 1] = 1.0

        damped_a, damped_b = capsule_contact_forces(
            first,
            second,
            radius_a=0.1,
            radius_b=0.1,
            stiffness=10.0,
            damping=2.0,
            velocity_a=velocity_a,
            velocity_b=velocity_b,
        )
        elastic_a, _ = capsule_contact_forces(
            first,
            second,
            radius_a=0.1,
            radius_b=0.1,
            stiffness=10.0,
        )
        self.assertGreater(np.linalg.norm(damped_a), np.linalg.norm(elastic_a))
        np.testing.assert_allclose(damped_a.sum(axis=0) + damped_b.sum(axis=0), [0.0, 0.0])


class SparseNeuralJacobianTests(unittest.TestCase):
    def test_sparse_bdf_jacobian_matches_dense_network_integration(self):
        network = CelegansWorm_NervousSystem(
            np.zeros((279, 279)), np.zeros((279, 279))
        )
        network.initcond = np.zeros_like(network.initcond)
        network.interp_i_Ext = lambda time: np.zeros(network.network_Size)
        sample_times = np.array([0.0, 0.01])
        dense = integrate.solve_ivp(
            network.forward_Network,
            (sample_times[0], sample_times[-1]),
            network.initcond,
            method="BDF",
            t_eval=sample_times,
            jac=network.forward_network_Jacobian,
            rtol=5e-6,
            atol=1e-3,
        )
        sparse = integrate.solve_ivp(
            network.forward_Network,
            (sample_times[0], sample_times[-1]),
            network.initcond,
            method="BDF",
            t_eval=sample_times,
            jac=lambda time, state: csc_matrix(
                network.forward_network_Jacobian(time, state)
            ),
            rtol=5e-6,
            atol=1e-3,
        )
        self.assertTrue(dense.success, dense.message)
        self.assertTrue(sparse.success, sparse.message)
        np.testing.assert_allclose(sparse.y, dense.y, rtol=1e-7, atol=1e-8)


class SolverConfigurationTests(unittest.TestCase):
    def test_stiff_body_uses_supported_vode_bdf_stepper(self):
        neural = SimpleNamespace(
            forward_Network=lambda time, state: -state,
            initcond=np.array([1.0]),
            timescale=0.01,
        )
        body = SimpleNamespace(
            forward_Body=lambda time, state: -state,
            initcond=np.array([1.0]),
        )

        neural_solver, body_solver = configure_ode_solver(
            neural, body, True, 1e-6, 1e-9, 1e-6, 1e-9
        )
        body_solver.integrate(0.01)

        self.assertTrue(neural_solver.successful())
        self.assertTrue(body_solver.successful())
        self.assertAlmostEqual(body_solver.y[0], np.exp(-0.01), places=5)

    def test_extreme_synaptic_sigmoid_inputs_remain_finite_without_overflow(self):
        size = 3
        network = SimpleNamespace(
            network_Size=size,
            neuron_Chemical=np.vstack((np.ones(size), np.ones(size), np.ones(size))),
        )
        voltage = np.array([-1e4, 0.0, 1e4])
        state = np.zeros(size)
        threshold = np.zeros(size)

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            activity = fwd_activity_Chemical(network, voltage, state, threshold)
            jacobian, _ = fwd_jacobian34_Linear(
                network, voltage, state, threshold
            )

        self.assertTrue(np.isfinite(activity).all())
        np.testing.assert_allclose(activity, [0.0, 0.5, 1.0])
        self.assertTrue(np.isfinite(jacobian).all())


class NumericalResultValidationTests(unittest.TestCase):
    def test_ivp_failure_incomplete_timing_and_nonfinite_states_raise(self):
        expected = np.array([0.0, 1.0])
        failed = SimpleNamespace(
            success=False,
            t=np.array([0.0]),
            y=np.array([[1.0]]),
            message="step limit reached",
        )
        with self.assertRaisesRegex(NumericalIntegrationError, "step limit reached"):
            validate_ivp_solution(failed, expected, 1, operation="test")

        incomplete = SimpleNamespace(
            success=True,
            t=np.array([0.0]),
            y=np.array([[1.0]]),
            message="success",
        )
        with self.assertRaisesRegex(NumericalIntegrationError, "incomplete"):
            validate_ivp_solution(incomplete, expected, 1, operation="test")

        mistimed = SimpleNamespace(
            success=True,
            t=np.array([0.0, 0.9]),
            y=np.array([[1.0, 0.5]]),
            message="success",
        )
        with self.assertRaisesRegex(NumericalIntegrationError, "mistimed"):
            validate_ivp_solution(mistimed, expected, 1, operation="test")

        nonfinite = SimpleNamespace(
            success=True,
            t=expected,
            y=np.array([[1.0, np.nan]]),
            message="success",
        )
        with self.assertRaisesRegex(NumericalIntegrationError, "non-finite"):
            validate_ivp_solution(nonfinite, expected, 1, operation="test")

    def test_legacy_ode_step_rejects_early_or_nonfinite_state(self):
        solver = SimpleNamespace(
            y=np.array([1.0]),
            t=0.5,
            successful=lambda: True,
        )
        with self.assertRaisesRegex(NumericalIntegrationError, "stopped early"):
            validate_ode_step(
                solver, 1.0, 1, operation="body", step_index=1
            )

        solver.t = 1.0
        solver.y = np.array([np.inf])
        with self.assertRaisesRegex(NumericalIntegrationError, "stopped early"):
            validate_ode_step(
                solver, 1.0, 1, operation="body", step_index=1
            )

    def test_network_and_body_solve_ivp_paths_raise_on_partial_results(self):
        partial = SimpleNamespace(
            success=True,
            t=np.array([0.0]),
            y=np.ones((1, 1)),
            message="success",
        )
        single_neuron = SimpleNamespace(
            network_Size=1,
            initcond=np.array([0.0]),
            forward_Network=lambda time, state: -state,
        )
        network_prep = SimpleNamespace(
            tf=1.0, nsteps=2, input_mat=np.zeros((2, 1))
        )
        with patch.object(network_simulations.integrate, "solve_ivp", return_value=partial):
            with self.assertRaises(NumericalIntegrationError):
                network_simulations.integrate_ode(
                    single_neuron, network_prep, 1e-6, 1e-9
                )

        body = SimpleNamespace(
            initcond=np.zeros(52),
            body_SegCount=24,
            forward_Body=lambda time, state: -state,
        )
        body_prep = SimpleNamespace(tf=1.0, nsteps=2)
        with patch.object(body_simulations.integrate, "solve_ivp", return_value=partial):
            with self.assertRaises(NumericalIntegrationError):
                body_simulations.integrate_ode(
                    body, body_prep, False, 1e-6, 1e-9
                )


class BodyEquationEquivalenceTests(unittest.TestCase):
    def test_vectorized_body_kinematics_matches_original_recurrence(self):
        body = CelegansWorm_MuscleBody(np.zeros((96, 279)))
        rng = np.random.default_rng(17)
        phi = rng.normal(size=body.body_SegCount)
        phidot = rng.normal(size=body.body_SegCount)
        x1, y1, xdot1, ydot1 = 1.2, -0.7, 0.4, -0.2

        expected = [np.zeros(body.body_SegCount) for _ in range(4)]
        x_ref, y_ref, xdot_ref, ydot_ref = expected
        x_ref[0], y_ref[0], xdot_ref[0], ydot_ref[0] = x1, y1, xdot1, ydot1
        for index in range(1, body.body_SegCount):
            previous = index - 1
            half = body.body_SegLength[index] / 2.0
            x_ref[index] = x_ref[previous] + half * (
                np.cos(phi[previous]) + np.cos(phi[index])
            )
            y_ref[index] = y_ref[previous] + half * (
                np.sin(phi[previous]) + np.sin(phi[index])
            )
            xdot_ref[index] = xdot_ref[previous] - half * (
                np.sin(phi[previous]) * phidot[previous]
                + np.sin(phi[index]) * phidot[index]
            )
            ydot_ref[index] = ydot_ref[previous] + half * (
                np.cos(phi[previous]) * phidot[previous]
                + np.cos(phi[index]) * phidot[index]
            )
        observed = fwd_body_Coordinates(
            body, x1, y1, xdot1, ydot1, phi, phidot
        )
        for actual, reference in zip(observed, expected):
            np.testing.assert_allclose(actual, reference, rtol=1e-13, atol=1e-13)

    def test_vectorized_coordinate_acceleration_matches_original_recurrence(self):
        body = CelegansWorm_MuscleBody(np.zeros((96, 279)))
        rng = np.random.default_rng(29)
        phi = rng.normal(size=body.body_SegCount)
        phidot = rng.normal(size=body.body_SegCount)
        phi_ddot = rng.normal(size=body.body_SegCount)
        wx = rng.normal(size=body.body_SegCount)
        wy = rng.normal(size=body.body_SegCount)

        expected_x = np.zeros(body.body_SegCount)
        expected_y = np.zeros(body.body_SegCount)
        for index in range(1, body.body_SegCount):
            half = body.body_SegLength[index] / 2.0
            left_x = np.cos(phi[index - 1]) * phidot[index - 1] ** 2 + np.sin(phi[index - 1]) * phi_ddot[index - 1]
            right_x = np.cos(phi[index]) * phidot[index] ** 2 + np.sin(phi[index]) * phi_ddot[index]
            left_y = np.sin(phi[index - 1]) * phidot[index - 1] ** 2 - np.cos(phi[index - 1]) * phi_ddot[index - 1]
            right_y = np.sin(phi[index]) * phidot[index] ** 2 - np.cos(phi[index]) * phi_ddot[index]
            expected_x[index] = expected_x[index - 1] - half * (left_x + right_x)
            expected_y[index] = expected_y[index - 1] - half * (left_y + right_y)

        actual_x, actual_y = fwd_body_D2Coordinates(
            body, phi, phidot, phi_ddot, wx, wy
        )
        mass_sum = np.sum(body.body_Mass)
        self.assertAlmostEqual(
            actual_x,
            np.reciprocal(mass_sum)
            * np.sum(body.body_SegLength * wx - body.body_Mass * expected_x),
            places=12,
        )
        self.assertAlmostEqual(
            actual_y,
            np.reciprocal(mass_sum)
            * np.sum(body.body_SegLength * wy - body.body_Mass * expected_y),
            places=12,
        )


class _IndependentTestNetwork:
    def __init__(self):
        self.network_Size = 279
        self.initcond = np.concatenate(
            (np.full(self.network_Size, -35.0), np.zeros(self.network_Size))
        )

    def compute_Vth(self, external_current):
        return np.zeros(self.network_Size)

    def forward_Network(self, time, state, contact_input_current=None):
        voltage, synaptic = np.split(state, 2)
        current = self.interp_i_Ext(time)
        if contact_input_current is not None:
            current = current + contact_input_current
        return np.concatenate(
            (-0.2 * (voltage + 35.0) + current, 0.01 * (voltage + 35.0) - 0.01 * synaptic)
        )

    def forward_network_Jacobian(self, time, state, contact_input_current=None):
        diagonal = np.concatenate(
            (-0.2 * np.ones(self.network_Size), -0.01 * np.ones(self.network_Size))
        )
        return np.diag(diagonal)


class SharedWorldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.body_template = CelegansWorm_MuscleBody(np.zeros((96, 279)))

    def make_agent(self, name, y_offset=0.0):
        network = _IndependentTestNetwork()
        body = copy.deepcopy(self.body_template)
        body.initcond[1] += y_offset
        return WormAgent(network, body, np.zeros(279), name)

    def test_contact_geometry_uses_body_force_application_coordinates(self):
        body = copy.deepcopy(self.body_template)
        rng = np.random.default_rng(12)
        state = body.initcond.copy()
        state[len(state) // 2 :] = rng.normal(size=len(state) // 2)
        q, qdot = np.split(state, 2)
        count = body.body_SegCount
        x, y, xdot, ydot = fwd_body_Coordinates(
            body,
            q[0],
            q[1],
            qdot[0],
            qdot[1],
            q[2 : 2 + count],
            qdot[2 : 2 + count],
        )

        centerline, velocity = _body_centerline_and_velocity(body, state)

        np.testing.assert_allclose(centerline, np.column_stack((x, y)))
        np.testing.assert_allclose(velocity, np.column_stack((xdot, ydot)))

    def test_two_noninteracting_worms_match_independent_runs(self):
        times = np.array([0.0, 0.01, 0.02])
        single = run_single_worm(
            self.make_agent("single"), times, mechanical_contact=False
        )["agents"][0]
        paired = run_two_worms(
            self.make_agent("first"),
            self.make_agent("second"),
            times,
            mechanical_contact=False,
            sensory_contact=False,
        )
        for key in ("v_solution", "s_solution", "muscle_force", "body_state"):
            np.testing.assert_allclose(
                paired["agents"][0][key], single[key], rtol=1e-10, atol=1e-10
            )
            np.testing.assert_allclose(
                paired["agents"][1][key], single[key], rtol=1e-10, atol=1e-10
            )
        for result in paired["agents"]:
            np.testing.assert_array_equal(
                result["contact_force"], np.zeros_like(result["contact_force"])
            )

    def test_separated_worms_have_zero_force_with_mechanics_enabled(self):
        world = run_two_worms(
            self.make_agent("separated-a"),
            self.make_agent("separated-b", y_offset=1.0),
            np.array([0.0, 0.01]),
            mechanical_contact=True,
            contact_stiffness=1.0,
            contact_radii=(0.05, 0.05),
        )
        for result in world["agents"]:
            np.testing.assert_array_equal(
                result["contact_force"], np.zeros_like(result["contact_force"])
            )

    def test_four_independent_worms_run_on_shared_clock(self):
        times = np.array([0.0, 0.01])
        agents = [
            self.make_agent("first", 0.0),
            self.make_agent("second", 1.0),
            self.make_agent("third", 2.0),
            self.make_agent("fourth", 3.0),
        ]
        world = run_shared_world(
            agents,
            times,
            mechanical_contact=True,
            contact_radii=(0.05, 0.05, 0.05, 0.05),
        )
        self.assertEqual(len(world["agents"]), 4)
        for result in world["agents"]:
            np.testing.assert_array_equal(
                result["time"], times
            )
            np.testing.assert_array_equal(
                result["contact_force"], np.zeros_like(result["contact_force"])
            )
        for first in range(len(world["agents"])):
            for second in range(first + 1, len(world["agents"])):
                self.assertFalse(
                    np.shares_memory(
                        world["agents"][first]["body_state"],
                        world["agents"][second]["body_state"],
                    )
                )

    def test_three_worm_contact_forces_accumulate_on_shared_agent(self):
        agents = [
            self.make_agent("center", 0.0),
            self.make_agent("upper", 0.05),
            self.make_agent("lower", -0.05),
        ]
        states = [agent.muscle_body.initcond.copy() for agent in agents]
        forces = _contact_forces(
            agents,
            states,
            stiffness=0.2,
            damping=0.0,
            radii=np.array([0.05, 0.05, 0.05]),
        )
        self.assertGreater(np.linalg.norm(forces[0]), 0.0)
        self.assertGreater(np.linalg.norm(forces[1]), 0.0)
        self.assertGreater(np.linalg.norm(forces[2]), 0.0)
        np.testing.assert_allclose(
            sum(force.sum(axis=0) for force in forces), [0.0, 0.0], atol=1e-12
        )

    def test_mechanical_contact_changes_integrated_body_motion(self):
        times = np.array([0.0, 0.01, 0.02])
        independent = run_two_worms(
            self.make_agent("free-a"),
            self.make_agent("free-b", 0.05),
            times,
            mechanical_contact=False,
            sensory_contact=False,
            contact_radii=(0.05, 0.05),
        )
        coupled = run_two_worms(
            self.make_agent("contact-a"),
            self.make_agent("contact-b", 0.05),
            times,
            mechanical_contact=True,
            contact_stiffness=0.2,
            contact_radii=(0.05, 0.05),
        )
        force = coupled["agents"][0]["contact_force"]
        self.assertGreater(np.max(np.linalg.norm(force, axis=2)), 0.0)
        body_change = np.linalg.norm(
            coupled["agents"][0]["body_state"][-1]
            - independent["agents"][0]["body_state"][-1]
        )
        self.assertGreater(body_change, 1e-8)

    def test_sensory_contact_is_separate_and_changes_touch_cell_activity(self):
        times = np.array([0.0, 0.01, 0.02])
        no_sensory = run_two_worms(
            self.make_agent("base-a"),
            self.make_agent("base-b", 0.05),
            times,
            mechanical_contact=False,
            sensory_contact=False,
            contact_radii=(0.05, 0.05),
        )
        sensory = run_two_worms(
            self.make_agent("sensory-a"),
            self.make_agent("sensory-b", 0.05),
            times,
            mechanical_contact=False,
            sensory_contact=True,
            contact_stiffness=0.2,
            contact_radii=(0.05, 0.05),
            sensory_gain_pA_per_force=1.0,
        )
        first, second = sensory["agents"]
        self.assertGreater(np.max(first["contact_sensory_current"]), 0.0)
        self.assertGreater(
            np.max(np.abs(first["v_solution"][:, 276] - no_sensory["agents"][0]["v_solution"][:, 276])),
            0.0,
        )
        non_touch = np.ones(279, dtype=bool)
        non_touch[[76, 80, 192, 196, 197, 227, 276, 278]] = False
        self.assertTrue(np.all(first["contact_sensory_current"][:, non_touch] == 0.0))
        self.assertFalse(np.shares_memory(first["v_solution"], second["v_solution"]))
        self.assertFalse(np.shares_memory(first["body_state"], second["body_state"]))

    def test_contact_force_integrates_into_original_body_rhs(self):
        body = copy.deepcopy(self.body_template)
        muscle = np.zeros(body.body_SegCount * 2)
        no_contact = body.forward_Body(0.0, body.initcond, muscle)
        nodal_force = np.zeros((body.body_SegCount, 2))
        nodal_force[:, 1] = 0.01
        with_contact = body.forward_Body(
            0.0, body.initcond, muscle, nodal_force
        )
        self.assertGreater(np.linalg.norm(with_contact - no_contact), 0.0)

        def center_of_mass_acceleration(derivative):
            state = body.initcond
            q, qdot = np.split(state, 2)
            qddot = derivative[len(q) :]
            count = body.body_SegCount
            phi = q[2 : 2 + count]
            phidot = qdot[2 : 2 + count]
            phi_ddot = qddot[2 : 2 + count]
            cos_term = np.cos(phi) * phidot**2 + np.sin(phi) * phi_ddot
            sin_term = np.sin(phi) * phidot**2 - np.cos(phi) * phi_ddot
            x_relative_acceleration = np.zeros(count)
            y_relative_acceleration = np.zeros(count)
            half_lengths = body.body_SegLength[1:] / 2.0
            x_relative_acceleration[1:] = np.cumsum(
                -half_lengths * (cos_term[:-1] + cos_term[1:])
            )
            y_relative_acceleration[1:] = np.cumsum(
                -half_lengths * (sin_term[:-1] + sin_term[1:])
            )
            total_mass = np.sum(body.body_Mass)
            return np.array(
                [
                    qddot[0]
                    + np.dot(body.body_Mass, x_relative_acceleration) / total_mass,
                    qddot[1]
                    + np.dot(body.body_Mass, y_relative_acceleration) / total_mass,
                ]
            )

        np.testing.assert_allclose(
            center_of_mass_acceleration(with_contact)
            - center_of_mass_acceleration(no_contact),
            nodal_force.sum(axis=0) / np.sum(body.body_Mass),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_repeated_contact_stays_finite_and_refinement_converges(self):
        coarse_times = np.array([0.0, 0.01, 0.02])
        fine_times = np.array([0.0, 0.005, 0.01, 0.015, 0.02])
        coarse = run_two_worms(
            self.make_agent("coarse-a"),
            self.make_agent("coarse-b", 0.05),
            coarse_times,
            contact_stiffness=0.1,
            contact_radii=(0.05, 0.05),
        )
        fine = run_two_worms(
            self.make_agent("fine-a"),
            self.make_agent("fine-b", 0.05),
            fine_times,
            contact_stiffness=0.1,
            contact_radii=(0.05, 0.05),
        )
        for result in coarse["agents"] + fine["agents"]:
            self.assertTrue(np.isfinite(result["body_state"]).all())
            self.assertTrue(np.isfinite(result["contact_force"]).all())
        np.testing.assert_allclose(
            coarse["agents"][0]["body_state"][-1],
            fine["agents"][0]["body_state"][-1],
            rtol=2e-3,
            atol=2e-3,
        )
        self.assertGreater(coarse["solver"]["body_nfev"], 0)

    def test_touch_proxy_uses_only_identified_mechanosensory_neurons(self):
        nodal_force = np.zeros((24, 2))
        nodal_force[20, 1] = 2.0
        current = contact_sensory_current(
            nodal_force, 279, gain_pA_per_force=0.5
        )
        self.assertGreater(current[276], 0.0)
        self.assertGreater(current[278], 0.0)
        self.assertEqual(np.count_nonzero(current), 2)
        with self.assertRaises(ValueError):
            contact_sensory_current(
                nodal_force, 279, gain_pA_per_force=-1.0
            )


if __name__ == "__main__":
    unittest.main()
