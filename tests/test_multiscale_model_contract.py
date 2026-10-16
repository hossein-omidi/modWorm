"""Fast Milestone-1 mathematical contract checks for actual modWorm modules.

Run from the repository root with:
    python -m unittest discover -s tests -p 'test_multiscale_model_contract.py' -v

These tests do not integrate or validate long-duration neural/body trajectories.
"""
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modWorm.serotonin.serotonin_modulation import SerotoninModulation, NEURON_INDICES
from modWorm.pdf_modulation import PDFR1Modulation
from modWorm import muscle_dynamics, network_interactions, utils


class MultiscaleModelContractTests(unittest.TestCase):
    def test_resting_threshold_lu_applies_row_permutation(self):
        network = SimpleNamespace(
            network_Size=2,
            neuron_Linear=np.array([[1., 1.], [1., 2.]]),
            network_Electrical=np.array([[0., 1.], [1., 0.]]),
            neuron_Chemical=np.ones((3, 2)),
            network_Chemical=np.zeros((2, 2, 2)),
        )
        network.network_Chemical[0, 0, 1] = -6.0
        network.network_Chemical[1] = 1.0
        network_interactions.init_vth_Linear(network)

        self.assertFalse(np.array_equal(network.P_vth, np.eye(2)))
        external = np.array([0.25, -0.5])
        expected = np.linalg.solve(
            network.P_vth @ network.LL_vth @ network.UU_vth,
            network.bb_vth - external,
        )
        actual = network_interactions.compute_vth_Linear(network, external)
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)

    def test_julia_threshold_rhs_applies_row_permutation(self):
        source = Path(network_interactions.__file__).read_text()
        self.assertIn("p.P_vth' * (p.bb_vth .- i_ext)", source)

    def test_cook_loader_rejects_missing_source_workbook(self):
        with patch.dict(os.environ):
            os.environ.pop("MODWORM_COOK_CONNECTOME", None)
            with self.assertRaisesRegex(FileNotFoundError, "Cook workbook is required"):
                utils.construct_connectome_Cook()

    def test_resting_threshold_includes_chemical_self_connections(self):
        network = SimpleNamespace(
            network_Size=1,
            neuron_Linear=np.array([[1.0], [-35.0]]),
            network_Electrical=np.zeros((1, 1)),
            neuron_Chemical=np.ones((3, 1)),
            network_Chemical=np.array([[[1.0]], [[1.0]]]),
        )
        network_interactions.init_vth_Linear(network)
        threshold = network_interactions.compute_vth_Linear(
            network, np.zeros(1)
        )

        equilibrium_gate = np.round(1.0 / (1.0 + 2.0), 4)
        expected = (-35.0 + equilibrium_gate) / (1.0 + equilibrium_gate)
        np.testing.assert_allclose(
            threshold, [expected], rtol=1e-13, atol=1e-13
        )

    def test_original_neuron_index_map_has_279_unique_entries(self):
        self.assertEqual(len(NEURON_INDICES), 279)
        self.assertEqual(set(NEURON_INDICES.values()), set(range(279)))

    def test_mod1_current_is_conductance_times_driving_force(self):
        model = SerotoninModulation(
            g_mod1=2.0, k_d=1.0, e_cl=-70.,
            concentration=5.0, target_neurons=('AIYL', 'AIYR')
        )
        v = np.full(279, -35., dtype=float)
        current = model.current_vector(v, 0.0)
        conductance = model.conductance_vector(0.0, 279)
        expected_g = 2.0 * 5.0 / 6.0
        for name in ('AIYL', 'AIYR'):
            idx = NEURON_INDICES[name]
            self.assertAlmostEqual(conductance[idx], expected_g)
            self.assertAlmostEqual(current[idx], expected_g * 35.0)
        self.assertEqual(np.count_nonzero(current), 2)
        h = 1e-4
        idx = NEURON_INDICES['AIYL']
        vp, vm = v.copy(), v.copy()
        vp[idx] += h
        vm[idx] -= h
        fd = (model.current_vector(vp, 0.)[idx] - model.current_vector(vm, 0.)[idx])/(2*h)
        self.assertAlmostEqual(fd, conductance[idx], places=8)

    def test_pdf_constant_input_matches_analytic_exponential_solution(self):
        pdf = PDFR1Modulation(concentration=5., tau_c=5., alpha=1.)
        self.assertGreater(len(pdf.target_indices), 0)
        target = pdf.alpha * pdf.receptor_occupancy(0.)
        initial = pdf.signal_state.copy()
        h = 0.8
        result = pdf.advance(h, 0.)
        expected = target + (initial - target)*np.exp(-h/pdf.tau_c)
        np.testing.assert_allclose(result, expected, rtol=1e-13, atol=1e-13)

    def test_pdf_two_half_steps_combine_exactly_for_constant_input(self):
        pdf = PDFR1Modulation(concentration=5., tau_c=5.)
        full = pdf.advance(0.8, 0.)
        pdf.signal_state = np.zeros_like(pdf.signal_state)
        pdf.advance(0.4, 0.)
        half = pdf.advance(0.4, 0.4)
        np.testing.assert_allclose(full, half, rtol=1e-13, atol=1e-13)

    def test_pdf_zero_default_gain_has_no_neural_current(self):
        pdf = PDFR1Modulation(concentration=5.)
        pdf.advance(1.0, 0.)
        self.assertTrue(np.all(pdf.hypothesis_current_vector(279) == 0.))

    def test_original_muscle_activation_is_algebraic(self):
        # Deliberately test the actual module functions without modifying the
        # muscle model or constructing an expensive neural/body ODE.
        m = np.zeros((96, 279))
        m[0, 0] = 1.0
        m[1, 1] = 1.0
        model = SimpleNamespace(muscle_Map=m, body_ForceScaling=6000.)
        v = np.zeros(279)
        v[0], v[1] = 0.01, 0.02
        vt = np.zeros(279)
        inputs = muscle_dynamics.fwd_muscle_Inputs(model, v-vt, single_step=True)
        ca = muscle_dynamics.fwd_muscle_Calcium(model, inputs, single_step=True)
        activity = muscle_dynamics.fwd_muscle_Activity(model, ca)
        force = muscle_dynamics.fwd_muscle_Force(model, activity)
        self.assertEqual(force.shape, (48,))
        np.testing.assert_allclose(force, 6000.*(15.*ca)**2/(1.+(15.*ca)**2))
        self.assertTrue(np.isfinite(force).all())

    def test_ppc_projection_uses_delayed_neural_voltage_not_body_stretch(self):
        # Projection via real implementation; a toy 2x279 muscle map keeps the
        # test fast while exercising sample-index and onset semantics exactly.
        m = np.zeros((2, 279))
        m[0, 0], m[1, 1] = 1., 1.
        mb = SimpleNamespace(muscle_Map=m, muscle_MapInv=np.linalg.pinv(m))
        net = SimpleNamespace(fdb_init_Int=118, fdb_delay_Int=60)
        v = np.zeros((119, 279))
        v[58, 0] = 3.0
        v[59, 0] = 19.0
        vt = np.zeros_like(v)
        dummy = np.zeros((119, 1))
        before = network_interactions.fwd_i_ProprioceptiveDelayed(
            net, mb, v, v, vt, dummy, dummy, 117
        )
        self.assertTrue(np.all(np.asarray(before) == 0.))
        onset = network_interactions.fwd_i_ProprioceptiveDelayed(
            net, mb, v, v, vt, dummy, dummy, 118
        )
        expected = np.zeros(279)
        expected[0] = 3.
        np.testing.assert_allclose(onset, expected, rtol=1e-12, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
