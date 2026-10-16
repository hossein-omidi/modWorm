# Apply Milestone 1 audit patch

**Target repository:** `hossein-omidi/modWorm`, **branch:** `dev-multiscale`  
**Reviewed HEAD:** `52a3e83ecd5f25e744cd14062c569646b16f2768`

This is a **small overlay ZIP**, not a new copy of the entire repository.

## Installation

1. Switch to `dev-multiscale` and back up or commit your local uncommitted changes first.
2. Extract the ZIP **into the repository root**, keeping the `docs/` and `tests/` directories. Allow replacement of `tests/test_multiscale_reference.py` after reviewing the diff.
3. Run `git diff -- tests/test_multiscale_reference.py` and review `docs/multiscale_milestone1_review.md`.
4. Run the focused contract tests:

   `python -m unittest discover -s tests -p 'test_multiscale_model_contract.py' -v`

5. Run the corrected numerical smoke suite when your environment can afford it:

   `python -m unittest discover -s tests -p 'test_multiscale_reference.py' -v`

Do **not** overwrite your `modWorm/` library or data directory. This ZIP modifies no model equations.

## Files inside

- `docs/multiscale_milestone1_review.md` — source-backed audit and known gaps.
- `tests/test_multiscale_model_contract.py` — new fast mathematical tests.
- `tests/test_multiscale_reference.py` — replacement for the preexisting smoke tests: corrected 0.01 s PPC clock, canonical data, reproducibility, and fail-loud assertions.

## Important qualifications

Milestone 1 is an audit/contract stage, **not** proof of stable multiscale long-run integration. The existing Milestone-2 reference-results report still needs fresh numerical evidence and corrected duration wording after running the corrected suite. Do not cite its estimated runtimes or short finite-state checks as a converged high-accuracy reference.

The provided tests have been syntax-checked locally, but the complete remote branch could not be executed in the delivery sandbox. Execute numerical tests against your own checked-out branch before treating any results as passing.
