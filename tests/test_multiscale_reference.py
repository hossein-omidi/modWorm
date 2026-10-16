"""Milestone 2 fast contracts + opt-in full-connectome reference checks.

Fast tests do not integrate worms. Enable full neural/body runs explicitly:
  MODWORM_RUN_SLOW_REFERENCES=1 python -m unittest discover -s tests -p test_multiscale_reference.py -v
  MODWORM_RUN_CONVERGENCE=1 python -m unittest discover -s tests -p test_multiscale_reference.py -v

The full-connectome cases require the Cook SI 5 workbook via
MODWORM_COOK_CONNECTOME and never use synthetic connectome fallbacks. The
checked-in Cook adjustment arrays are deltas, not complete connectomes. No
test claims that finite output implies convergence.
"""

from __future__ import annotations

from dataclasses import replace
import os
import unittest
from unittest.mock import patch

import numpy as np

from modWorm import reference_integration as reference


class ReferenceInterfaceTests(unittest.TestCase):
    def test_configuration_has_correct_physical_time(self):
        settings = reference.ReferenceConfig(duration=0.02, steps=2, worms=2)
        self.assertAlmostEqual(settings.dt, 0.01)
        np.testing.assert_allclose(settings.times, [0.0, 0.01, 0.02])

    def test_invalid_configuration_is_rejected(self):
        for kwargs in (
            {"duration": 0}, {"steps": 0}, {"worms": 0},
            {"steps": 0.5}, {"neural_rtol": 0},
            {"contact_stiffness": -1.0}, {"pdf_uM": -1.0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                reference.ReferenceConfig(**kwargs)

    def test_scaled_comparison_is_dimensionless_and_validated(self):
        residual = reference.normalized_difference(
            [1.0, -2.0], [1.0, -2.1], atol=0.1, rtol=0.01
        )
        self.assertAlmostEqual(residual["max"], 0.1 / (0.1 + 0.01 * 2.1))
        self.assertGreaterEqual(residual["max"], residual["rms"])
        with self.assertRaises(ValueError):
            reference.normalized_difference([1.0, 2.0], [1.0], atol=0.1, rtol=0.01)
        with self.assertRaises(ValueError):
            reference.normalized_difference([float("nan")], [0.0], atol=0.1, rtol=0.01)

    def test_refinement_reuses_world_entrypoint_and_aligns_observations(self):
        def fake_world(agents, times, **kwargs):
            # Test only the orchestration: a synthetic error O(H), not a worm model.
            h = times[1] - times[0]
            return {
                "time": times.copy(),
                "agents": [{
                    "name": agents[0].name,
                    "v_solution": (times + h)[:, None],
                    "s_solution": np.zeros((len(times), 1)),
                    "body_state": np.zeros((len(times), 2)),
                }],
                "solver": {"neural": "synthetic mock", "body": "synthetic mock"},
            }

        from modWorm.shared_world import WormAgent
        from types import SimpleNamespace

        def fake_factory(config):
            return [WormAgent(SimpleNamespace(), SimpleNamespace(), np.zeros((len(config.times), 1)), "worm_0")]

        with patch.object(reference.shared_world, "run_shared_world", fake_world):
            results = reference.run_refinement(
                reference.ReferenceConfig(duration=0.02, steps=2),
                agent_factory=fake_factory,
            )
        self.assertEqual(results["dt"], [0.01, 0.005, 0.0025])
        e1 = results["coarse_vs_medium"]["worm_0"]["v_solution"]["max"]
        e2 = results["medium_vs_fine"]["worm_0"]["v_solution"]["max"]
        self.assertGreater(e1, e2)
        self.assertGreater(e2, 0)

    def test_ppc_refinement_rejects_index_delay_semantics(self):
        with self.assertRaisesRegex(ValueError, "PPC delay"):
            reference.run_refinement(reference.ReferenceConfig(ppc=True))


@unittest.skipUnless(os.environ.get("MODWORM_RUN_SLOW_REFERENCES") == "1",
                     "Full 279-neuron/body smoke tests are opt-in: MODWORM_RUN_SLOW_REFERENCES=1")
class FullModelSmokeTests(unittest.TestCase):
    def test_actual_cook_model_baseline(self):
        config = reference.ReferenceConfig(duration=0.01, steps=1)
        run = reference.run_reference(config)
        self.assertEqual(run["result"]["solver"]["neural"], "BDF")
        self.assertEqual(run["result"]["solver"]["body"], "RK45")
        sample = run["result"]["agents"][0]
        self.assertEqual(sample["v_solution"].shape, (2, 279))
        self.assertEqual(sample["s_solution"].shape, (2, 279))
        self.assertEqual(sample["body_state"].shape, (2, 52))
        for field in ("v_solution", "s_solution", "muscle_force", "body_state"):
            self.assertTrue(np.isfinite(sample[field]).all(), field)

    def test_identical_initial_states_across_factory_runs(self):
        config = reference.ReferenceConfig(duration=0.01, steps=1, worms=2)
        a = reference.make_full_model_agents(config)
        b = reference.make_full_model_agents(replace(config, steps=2))
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x.nervous_system.initcond, y.nervous_system.initcond)
            np.testing.assert_array_equal(x.muscle_body.initcond, y.muscle_body.initcond)
        self.assertIsNot(a[0].nervous_system, a[1].nervous_system)
        self.assertIsNot(a[0].muscle_body, a[1].muscle_body)


@unittest.skipUnless(os.environ.get("MODWORM_RUN_CONVERGENCE") == "1",
                     "True full-model step refinement is expensive: MODWORM_RUN_CONVERGENCE=1")
class FullModelRefinementTests(unittest.TestCase):
    def test_smooth_baseline_step_refinement(self):
        report = reference.run_refinement(reference.ReferenceConfig(duration=0.01, steps=1))
        for comparison in ("coarse_vs_medium", "medium_vs_fine"):
            for name, metrics in report[comparison].items():
                for key, errors in metrics.items():
                    self.assertTrue(np.isfinite(errors["max"]), (name, key))
                    self.assertTrue(np.isfinite(errors["rms"]), (name, key))
        # Finite differences do not prove a converged numerical reference.


@unittest.skipUnless(os.environ.get("MODWORM_RUN_MULTI_WORM_REFERENCES") == "1",
                     "Two full Cook-network worms are expensive: MODWORM_RUN_MULTI_WORM_REFERENCES=1")
class MultiwormReferenceTests(unittest.TestCase):
    def test_first_worm_agrees_with_isolated_simulation_when_decoupled(self):
        single = reference.run_reference(reference.ReferenceConfig(duration=0.01, steps=1, worms=1))
        pair = reference.run_reference(reference.ReferenceConfig(duration=0.01, steps=1, worms=2))
        a = single["result"]["agents"][0]
        b = pair["result"]["agents"][0]
        for key in ("v_solution", "s_solution", "muscle_force", "body_state"):
            np.testing.assert_allclose(a[key], b[key], rtol=1e-5, atol=1e-3,
                                       err_msg=f"decoupled worm mismatch: {key}")


if __name__ == "__main__":
    unittest.main()
