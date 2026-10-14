import unittest
import warnings
from types import SimpleNamespace

import numpy as np

from modWorm.behavior_analysis import locomotion_metrics, population_metrics
from modWorm.collective import capsule_contact_forces
from modWorm.network_interactions import fwd_jacobian12_Linear, fwd_jacobian34_Linear
from modWorm.predefined_classes_nv import CelegansWorm_NervousSystem
from modWorm.pdf_modulation import PDFR1Modulation
from modWorm.network_dynamics import fwd_activity_Chemical
from modWorm.proprioception_simulation import configure_ode_solver


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

        no_force_a, no_force_b = capsule_contact_forces(
            first,
            separated,
            radius_a=0.1,
            radius_b=0.1,
            stiffness=10.0,
        )
        np.testing.assert_array_equal(no_force_a, np.zeros_like(first))
        np.testing.assert_array_equal(no_force_b, np.zeros_like(separated))

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


if __name__ == "__main__":
    unittest.main()
