import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from modWorm import network_dynamics, network_interactions
from modWorm.predefined_classes_nv import CelegansWorm_NervousSystem
from modWorm.serotonin import NEURON_INDICES, SerotoninModulation


class SerotoninModulationTests(unittest.TestCase):
    def test_hill_activation_and_current_sign(self):
        modulation = SerotoninModulation(concentration=5.0)

        self.assertAlmostEqual(modulation.receptor_activation(0.0), 5.0 / 6.0)
        self.assertAlmostEqual(modulation.current(-35.0, 0.0), 2.0 * (5.0 / 6.0) * 35.0)
        self.assertEqual(
            SerotoninModulation(concentration=0.0).receptor_activation(0.0),
            0.0,
        )

    def test_hill_dose_response_is_monotone_and_saturating(self):
        modulation = SerotoninModulation()
        doses = [0.0, 0.1, 1.0, 5.0, 1e6]
        responses = []
        for dose in doses:
            modulation.set_concentration(dose)
            responses.append(modulation.receptor_activation(0.0))

        self.assertEqual(responses[0], 0.0)
        self.assertTrue(np.all(np.diff(responses) > 0.0))
        self.assertLess(responses[-1], 1.0)
        self.assertGreater(responses[-1], 0.999)

    def test_time_varying_control_and_disabled_layer(self):
        modulation = SerotoninModulation(concentration=lambda time: time * 2.0)

        self.assertAlmostEqual(modulation.concentration_at(2.5), 5.0)
        modulation.enabled = False
        self.assertEqual(modulation.receptor_activation(2.5), 0.0)

    def test_only_configured_neurons_receive_current(self):
        modulation = SerotoninModulation(concentration=5.0)
        voltage = np.full(279, -35.0)

        current = modulation.current_vector(voltage, 0.0)

        non_targets = np.ones(279, dtype=bool)
        non_targets[[133, 137]] = False
        self.assertTrue(np.all(current[non_targets] == 0.0))
        self.assertGreater(current[133], 0.0)
        self.assertGreater(current[137], 0.0)

    def test_optional_rif_and_asi_targets_are_configurable(self):
        modulation = SerotoninModulation(
            concentration=5.0,
            target_neurons=("RIFR", "ASIL"),
        )
        current = modulation.current_vector(np.full(279, -35.0), 0.0)

        non_targets = np.ones(279, dtype=bool)
        non_targets[[75, 148]] = False
        self.assertTrue(np.all(current[non_targets] == 0.0))
        self.assertGreater(current[75], 0.0)
        self.assertGreater(current[148], 0.0)

    def test_zero_concentration_is_a_noop_for_networks_without_targets(self):
        modulation = SerotoninModulation(concentration=0.0)

        np.testing.assert_array_equal(
            modulation.current_vector(np.array([-35.0]), 0.0),
            np.zeros(1),
        )
        np.testing.assert_array_equal(
            modulation.conductance_vector(0.0, 1),
            np.zeros(1),
        )

    def test_target_indices_match_the_connectome_neuron_list(self):
        neurons_path = Path(__file__).parents[1] / "modWorm" / "neurons.json"
        with neurons_path.open(encoding="utf-8") as neurons_file:
            neurons = json.load(neurons_file)["neurons"]
        indices_by_name = {neuron["name"]: neuron["index"] for neuron in neurons}

        for name, index in NEURON_INDICES.items():
            self.assertEqual(indices_by_name[name], index)
        self.assertNotIn("NSM", indices_by_name)

    def test_zero_concentration_produces_no_modulatory_current(self):
        voltage = np.linspace(-60.0, 10.0, 279)
        network = SimpleNamespace()

        for modulation in (
            SerotoninModulation(concentration=0.0),
            SerotoninModulation(concentration=5.0, enabled=False),
        ):
            network.serotonin_modulation = modulation
            current = network_dynamics.fwd_i_Serotonin(network, voltage, 7.5)
            np.testing.assert_array_equal(current, np.zeros(279))

    def test_network_rhs_subtracts_current_only_from_aiy(self):
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
        network.serotonin_modulation = SerotoninModulation(concentration=5.0)

        modulated = network.forward_Network(0.0, state)

        expected = np.zeros(size)
        expected[[133, 137]] = (
            -network.serotonin_modulation.current_vector(voltage, 0.0)[[133, 137]]
        )
        np.testing.assert_allclose(modulated[:size] - baseline[:size], expected)
        np.testing.assert_array_equal(modulated[size:], baseline[size:])

        network.serotonin_modulation.set_concentration(0.0)
        zero_serotonin = network.forward_Network(0.0, state)
        np.testing.assert_array_equal(zero_serotonin, baseline)

    def test_current_changes_only_aiy_voltage_jacobian_entries(self):
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
        baseline, _ = network_interactions.fwd_jacobian12_Linear(
            network, voltage, synaptic_state
        )
        network.serotonin_modulation = SerotoninModulation(concentration=5.0)

        modulated, _ = network_interactions.fwd_jacobian12_Linear(
            network, voltage, synaptic_state, 0.0
        )

        expected = np.zeros(size)
        expected[[133, 137]] = -(2.0 * (5.0 / 6.0))
        np.testing.assert_allclose(np.diag(modulated - baseline), expected)
        off_diagonal = modulated - baseline
        np.fill_diagonal(off_diagonal, 0.0)
        self.assertTrue(np.all(off_diagonal == 0.0))


if __name__ == "__main__":
    unittest.main()
