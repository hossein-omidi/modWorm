# Apply the Milestone 2 reference patch to `dev-multiscale`

**Target GitHub branch snapshot:** `52a3e83ecd5f25e744cd14062c569646b16f2768`.
The user's local working tree may contain previous uncommitted patches; review
diffs before overwriting anything. This archive is an **overlay**, not a full
repository checkout. No source neural/body equations are replaced.

## Files in this ZIP

| Relative path | Action |
|---|---|
| `modWorm/reference_integration.py` | Add: thin integration/refinement interface using existing `shared_world` |
| `tests/test_multiscale_reference.py` | Replace: fast reference contracts, opt-in slow real-model tests |
| `examples/reference_integration.py` | Replace: working CLI scenario runner |
| `examples/diagnose_reference_neural_solvers.py` | Add: optional VODE/BDF + Jacobian diagnosis |
| `docs/multiscale_reference_results.md` | Replace: corrected evidence-based reference protocol/status |
| `docs/multiscale_audit_addendum.md` | Add: source audit and semantics clarifications |

Do **not** delete the existing neural, body, muscle, modulation, shared-world,
collective, PPC, Milestone 1 specifications, or mathematical contract tests.

## Windows PowerShell: recommended usage

1. Check out your `dev-multiscale` branch and back up uncommitted changes.
2. Inspect ZIP entries, then extract into the repository root while preserving
   the folder hierarchy. Replacements are limited to the files shown above.
3. Ensure your normal `celegan` Conda environment is active.
4. Run the quick, non-expensive interface suite:

   `python -m unittest discover -s tests -p "test_multiscale_reference.py" -v`

   Expected: the fast tests run, while the actual full neural/body runs are
   clearly marked **skipped** unless you enable them. Skipping them is not
   proof the full model works.

5. Run the existing Milestone 1 contract suite:

   `python -m unittest discover -s tests -p "test_multiscale_model_contract.py" -v`

6. Run one genuine canonical Cook-network/BDF/RK45 smoke test in isolation:

   `$env:MODWORM_RUN_SLOW_REFERENCES="1"`

   `python -m unittest discover -s tests -p "test_multiscale_reference.py" -v -f`

   `Remove-Item Env:MODWORM_RUN_SLOW_REFERENCES`

   This is a real 279-neuron/52-body-state experiment. It may take appreciable
   time. Stop with Ctrl+C if needed and share the traceback, not just the last
   line printed.

Optionally validate two independent full-network agents:

   `$env:MODWORM_RUN_MULTI_WORM_REFERENCES="1"`

   `python -m unittest discover -s tests -p "test_multiscale_reference.py" -v -f`

   `Remove-Item Env:MODWORM_RUN_MULTI_WORM_REFERENCES`

7. Run a single real scenario explicitly:

   `python examples/reference_integration.py --scenario baseline --duration 0.01 --steps 1`

   Run `python examples/reference_integration.py --help` for options.

   PDF gain and mechanosensory transduction are explicitly speculative and
   default to zero. To activate a sensory-contact hypothesis, you must set
   a documented positive `--sensory-gain`; there is no silently invented gain.

8. Only if the single run succeeds, enable the measured step-refinement study:

   `$env:MODWORM_RUN_CONVERGENCE="1"`

   `python -m unittest discover -s tests -p "test_multiscale_reference.py" -v -f`

   `Remove-Item Env:MODWORM_RUN_CONVERGENCE`

   Or explicitly:

   `python examples/reference_integration.py --scenario baseline --duration 0.01 --steps 1 --refine --out results/reference_baseline.json`

Optionally inspect the unresolved solver failure directly:

   `python examples/diagnose_reference_neural_solvers.py --duration 0.01`

   `python examples/diagnose_reference_neural_solvers.py --duration 0.01 --legacy`

   The `--legacy` option can be slow and repeats the original VODE settings.

The real model may still be slow; do not interpret this patch as a performance
breakthrough or a completed biological validation.

## Important constraints and known issues

- `shared_world` uses SciPy BDF/RK45. The legacy
  `proprioception_simulation.run_network` still uses VODE and has a recorded
  'excess work' failure. This patch **does not** alter or conceal that path.
- Do not use `--refine` with `--scenario ppc`. The original PPC uses index-based
  delay and cannot be physically equated across different macro timesteps.
- PDFR-1's neural bias defaults to zero. 'PDF' mode checks signaling, not a
  proven PDF-driven locomotion mechanism.
- Contact force units, sensory gains, and substrate locomotion remain
  scientifically uncalibrated.
- A short successful simulation is a smoke test. True numerical convergence
  requires multiple refinements, error analyses, and external verification.
- The `--refine` study holds original model parameters and initial values
  fixed; external stimulation is zero by default, preventing mismatched
  interpolation functions across grids. Add controlled time-dependent input
  profiles only with physically matched event times and values.
- The patch uses installed NumPy/SciPy and the project's original import
  interfaces; no Julia, new parallel framework, or rewritten neural ODE is
  required.

## Recommended next step

First share results for the real single-worm baseline; if that runs and shows
consistent refinement, proceed to the PDF/delay compatibility milestone. If
it fails, diagnose the precise neural, contact, or body cause before pursuing
additional numerical complexity.
