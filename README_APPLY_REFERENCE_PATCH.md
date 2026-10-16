# Targeted patch for the modWorm reference-test VODE failure

**Branch:** `dev-multiscale`  
**Triggered by:** `NumericalIntegrationError` at neural sample 1 (expected 0.01 s, reached ~0.001642 s).  
**Scope:** test / diagnostic / documentation only. NO changes to `modWorm/` numerical or biological modules.

## Contents

- `tests/test_multiscale_reference.py` — replaces the six premature legacy-VODE reference smoke tests with six **shared-world BDF smoke tests**. They are not a converged numerical reference.
- `examples/diagnose_legacy_vode.py` — **opt-in diagnostic** comparing the same neural first interval under legacy VODE and `solve_ivp(BDF)`; optional VODE internal-work budget experiment.
- `docs/reference_vode_failure_diagnosis.md` — evidence, differential diagnosis, limitations, and prerequisites for Milestone 2.

## Apply

1. Verify `git status`; back up local changes.
2. Extract this ZIP into the repository root, preserving `tests/`, `examples/`, and `docs/` directories. **Review the diff before accepting replacement** of `tests/test_multiscale_reference.py`.
3. Run the six smoke tests:

   `python -m unittest discover -s tests -p "test_multiscale_reference.py" -v -f`

4. Independently diagnose the legacy solver:

   `python examples/diagnose_legacy_vode.py`

   `python examples/diagnose_legacy_vode.py --extra-vode` (optional, potentially expensive).

5. Retain your earlier seven mathematical-contract tests:

   `python -m unittest discover -s tests -p "test_multiscale_model_contract.py" -v`

## Interpretation

A successful shared-world test is NOT a fix to `proprioception_simulation.run_network`. If the legacy solver must remain supported, investigate the collected diagnostics and provide an explicit validated repair in a separate change. Do not blindly increase VODE's `nsteps` or weaken solver-failure checks.

## Validation performed in the delivery environment

- Python syntax compilation (yes).
- ZIP integrity and content manifest (yes).
- End-to-end full-connectome modWorm tests (not run; full repository/data are not locally available).
