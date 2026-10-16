"""Julia-independent regression tests for the PyArray/Sundials bridge patch.

Run as `python -m unittest discover -s tests -p 'test_julia_native_bridge_patch.py' -v`.
The tests are intentionally cheap; they do not validate the Julia runtime.
"""
from pathlib import Path
import importlib.util
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'apply_julia_native_arrays.py'

spec = importlib.util.spec_from_file_location('modworm_julia_patch_script', SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

FIXTURE = '''from modWorm import Main


def run_network_julia_ensemble():
    return Main.run_network_ppc_ensemble(vars_NN_nv_ens, vars_NN_mb_ens, verbose_period, rtol_nv, atol_nv, rtol_mb, atol_mb)


def run_network_julia():
    return Main.run_network_ppc(vars_NN_nv, vars_NN_mb, verbose_period, rtol_nv, atol_nv, rtol_mb, atol_mb)
'''


class NativeJuliaBridgePatchTests(unittest.TestCase):
    def test_known_call_sites_patched_once_and_kept_valid(self):
        patched, changed = mod.patch_source(FIXTURE)
        self.assertTrue(changed)
        self.assertIn('modworm_native_ppc_dict', patched)
        self.assertIn('Array(value)', patched)
        self.assertIn('Main.modworm_native_ppc_dict(vars_NN_nv)', patched)
        self.assertIn('Main.modworm_native_ppc_dict(vars_NN_mb)', patched)
        self.assertIn('modworm_native_ppc_dict_list(vars_NN_nv_ens)', patched)
        self.assertIn('modworm_native_ppc_dict_list(vars_NN_mb_ens)', patched)
        compile(patched, 'fixture.py', 'exec')
        repeated, changed_again = mod.patch_source(patched)
        self.assertFalse(changed_again)
        self.assertEqual(repeated, patched)

    def test_unknown_upstream_file_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, 'anchor'):
            mod.patch_source(FIXTURE.replace('run_network_ppc(', 'renamed_ppc('))

    def test_ambiguous_call_sites_fail_closed(self):
        with self.assertRaisesRegex(RuntimeError, 'exactly one'):
            mod.patch_source(FIXTURE + '\n' + mod.SINGLE_ORIGINAL + '\n')

    def test_new_backend_converts_before_ode_problem(self):
        julia = (ROOT / 'modWorm' / '_julia_modulated_ppc.jl').read_text(encoding='utf-8')
        first = julia.index('modworm_native_ppc_dict(raw_nn)')
        ode = julia.index('neural_problem = ODEProblem(')
        self.assertLess(first, ode)
        self.assertIn('nn.initcond isa Vector{Float64}', julia)
        self.assertIn('mb.initcond isa Vector{Float64}', julia)
        self.assertIn('CVODE_BDF()', julia)
        self.assertIn('integrate_ode_ppc(', julia)


if __name__ == '__main__':
    unittest.main()
