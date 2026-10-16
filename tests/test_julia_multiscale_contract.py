"""Fast, Julia-independent mathematical and interface tests.

Run from the repository root with:
  python -m unittest discover -s tests -p 'test_julia_multiscale_contract.py' -v
"""
import importlib.util
from pathlib import Path
import unittest
from types import SimpleNamespace

import numpy as np

MODULE = Path(__file__).resolve().parents[1] / 'modWorm' / 'julia_multiscale.py'
spec = importlib.util.spec_from_file_location('julia_multiscale_standalone', MODULE)
module = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class StubPDF:
    def __init__(self, *, concentration=3.0, tau_c=4.0, alpha=0.7, enabled=True):
        self.concentration = concentration
        self.tau_c = tau_c
        self.alpha = alpha
        self.enabled = enabled
        self.target_indices = (0, 2)
        self.signal_state = np.array([0.15, 0.29])
        self.coupling_gain_pA_per_signal = 1.0

    def receptor_occupancy(self, t):
        return (self.concentration / (1 + self.concentration)) if self.enabled else 0.0


class StubSerotonin:
    enabled = True
    target_indices = (1, 3)
    g_mod1 = 2.0
    e_cl = -70.0

    def receptor_activation(self, t):
        return 0.25


class JuliaMathematicsTests(unittest.TestCase):
    def test_exact_pdf_matches_analytic_solution_for_unequal_initial_targets(self):
        p = StubPDF()
        t = np.linspace(0, 1.25, 6)
        knots, target = module._pdf_profile(p, t)
        q = p.alpha * p.receptor_occupancy(0.0)
        expected = q + (p.signal_state - q) * np.exp(-t[:, None] / p.tau_c)
        np.testing.assert_allclose(knots, expected, atol=1e-14)
        np.testing.assert_allclose(target, q)

    def test_pdf_step_at_piecewise_input_jump_is_causal(self):
        p = StubPDF()
        p.receptor_occupancy = lambda t: 0.0 if t < 0.4 else 0.6
        t = np.array([0.0, 0.2, 0.4, 0.7])
        knots, eq = module._pdf_profile(p, t)
        self.assertAlmostEqual(eq[0], 0.0)
        self.assertAlmostEqual(eq[1], 0.0)
        self.assertAlmostEqual(eq[2], p.alpha * 0.6)
        self.assertAlmostEqual(knots[1, 0], p.signal_state[0]*np.exp(-.2/p.tau_c))

    def test_full_neuron_target_masks_and_one_based_indices(self):
        nn = SimpleNamespace(network_Size=279)
        t = np.array([0.0, .01, .02])
        cfg = module._configure_modulation(nn, t, StubSerotonin(), StubPDF())
        expected = np.zeros(279)
        expected[[1, 3]] = 2.0
        np.testing.assert_array_equal(cfg['modworm_g_mod1'], expected)
        np.testing.assert_array_equal(cfg['modworm_pdf_targets'], [1, 3])
        np.testing.assert_array_equal(cfg['modworm_5ht_activation'], 0.25)

    def test_default_no_modulation_has_zero_currents(self):
        nn = SimpleNamespace(network_Size=279)
        cfg = module._configure_modulation(nn, np.array([0.0, .01]), None, None)
        self.assertEqual(cfg['modworm_pdf_gain'], 0.0)
        self.assertEqual(np.count_nonzero(cfg['modworm_g_mod1']), 0)
        self.assertEqual(cfg['modworm_pdf_knots'].shape, (2, 0))

    def test_settings_reject_invalid_tolerances(self):
        with self.assertRaises(ValueError):
            module.JuliaRunSettings(neural_rtol=0)
        with self.assertRaises(ValueError):
            module.JuliaRunSettings(body_atol=-1)

    def test_inputs_follow_model_timescale(self):
        nn = SimpleNamespace(network_Size=279, timescale=.01)
        inputs, t = module._validate_inputs(nn, np.zeros((4, 279)))
        np.testing.assert_allclose(t, [0, .01, .02, .03])
        self.assertTrue(inputs.flags.c_contiguous)
        with self.assertRaises(ValueError):
            module._validate_inputs(nn, np.zeros((4, 278)))

    def test_julia_rhs_sign_and_original_function_reuse_are_present(self):
        text = MODULE.with_name('_julia_modulated_ppc.jl').read_text()
        for symbol in ('fwd_i_Linear', 'fwd_i_Electrical', 'fwd_i_Chemical',
                       'fwd_activity_Chemical', 'construct_NN_ppc',
                       'integrate_ode_ppc', 'CVODE_BDF', 'forward_Body!'):
            self.assertIn(symbol, text)
        self.assertIn('i_mod1 .+ i_pdfr1', text)


class IntegrationContractMockTests(unittest.TestCase):
    """Mocks verify Python-to-Julia data wiring; not a Julia runtime test."""
    def test_run_preserves_model_parameters_and_commits_pdf_after_success(self):
        from unittest.mock import patch
        from types import ModuleType

        class MockPPC_Julia:
            network_Size = 279
            timescale = .01
            initcond = np.zeros(558)
            neuron_Linear = np.vstack((np.ones(279), -35.0*np.ones(279)))
            network_Electrical = np.zeros((279,279))
            network_Chemical = np.zeros((2,279,279))
            neuron_Chemical = np.vstack((np.ones(279),np.ones(279),np.ones(279)))
            LL_vth = np.eye(279)
            UU_vth = -np.eye(279)
            P_vth = np.eye(279)
            bb_vth = -35*np.ones(279)
            def forward_Network(self): pass
            def compute_Vth(self): pass
            def proprioceptive_FDB(self): pass
        class MockBodyPPC_Julia:
            body_SegCount = 24
            initcond = np.zeros(52)
            def forward_Muscles(self): pass
            def forward_Body(self): pass
        class FakeMain:
            def modworm_run_modulated_ppc(self, nn, mb, *args):
                self.nn = nn
                self.mb = mb
                return {
                    'x0': np.zeros(2), 'y0': np.zeros(2),
                    'phi': np.zeros((2,24)),
                    'v_solution': np.zeros((2,279)),
                    's_solution': np.zeros((2,279)),
                    'v_threshold': np.zeros((2,279)),
                    'muscle_force': np.zeros((2,48)),
                }
        mock_main = FakeMain()
        fake_body_sim = ModuleType('body_simulations')
        fake_body_sim.solve_xy = lambda body, x0, y0, phi: (np.zeros((2,25)), np.zeros((2,25)))
        fake_body_sim.postprocess_xy = lambda body, x, y: (x,y)
        fake_modworm = ModuleType('modWorm')
        fake_modworm.body_simulations = fake_body_sim
        pdf = StubPDF()
        start = pdf.signal_state.copy()
        with patch.object(module, '_register_julia', return_value=mock_main), \
             patch.dict(sys.modules, {'modWorm':fake_modworm, 'modWorm.body_simulations':fake_body_sim}):
            out=module.run_modulated_julia(
                MockPPC_Julia(), MockBodyPPC_Julia(), np.zeros((2,279)),
                serotonin=StubSerotonin(), pdfr1=pdf)
        self.assertEqual(out['v_solution'].shape,(2,279))
        self.assertEqual(out['x_solution'].shape,(2,25))
        self.assertEqual(out['pdf_signal_solution'].shape,(2,2))
        self.assertEqual(mock_main.nn['modworm_pdf_targets'].tolist(),[1,3])
        self.assertEqual(mock_main.mb['fR'].shape,(23,))
        self.assertFalse(np.array_equal(pdf.signal_state,start))
        self.assertAlmostEqual(out['time'][1],.01)

class PopulationValidationTests(unittest.TestCase):
    def test_shared_neural_or_pdf_instances_rejected_before_julia_initialization(self):
        class FakeModel:
            network_Size = 279
            timescale = .01
        same = FakeModel()
        other = FakeModel()
        pdf = StubPDF()
        x = np.zeros((2,279))
        with self.assertRaisesRegex(ValueError, 'neural model'):
            module.run_independent_julia_population([(same,object(),x,None,pdf),
                                                     (same,object(),x,None,StubPDF())])
        with self.assertRaisesRegex(ValueError, 'PDF'):
            module.run_independent_julia_population([(same,object(),x,None,pdf),
                                                     (other,object(),x,None,pdf)])


if __name__ == '__main__':
    unittest.main()
