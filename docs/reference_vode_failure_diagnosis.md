# Milestone 1 → 2: reference integration failure triage

## Observed result (provided by the user)

- `test_multiscale_model_contract.py`: 7/7 tests passed.
- Original `test_multiscale_reference.py` baseline failed at the first neural step.
- `scipy.integrate.ode` VODE/BDF warning: `Excess work done on this call. (Perhaps wrong MF.)`.
- Expected sample time: `0.01 s`; VODE stopped at `0.0016421836381860318 s`.
- Neural state dimension: 558, matching the basic 279-voltage + 279-synaptic configuration.
- The failure originated inside `proprioception_simulation.run_network` → `integrate_ode` → `validate_ode_step`, **before a body step completed**.

## What is established and what is not

Confirmed: the legacy neural solver exhausted its internal work allowance or otherwise returned an excess-work failure before target time. Its default internal work budget is 500 steps per requested integration call. This is not evidence that body equations, receptor equations, or the neural Jacobian are necessarily wrong. Neither the correct fix nor accuracy of an alternative solver is established by this traceback alone.

The notebook and the earlier diagnostic loaded `conn_gap_adjust_Cook.npy` and `conn_syn_adjust_Cook.npy` directly. These files are adjustment deltas, not complete adjacency matrices. With those deltas alone the initial neural Jacobian had a positive real eigenvalue of about `5.04e4 s^-1`. Reconstructing the full Cook matrices from the SI 5 workbook and then applying the deltas produced nonnegative connectome entries and a maximum initial Jacobian real part of about `-3.50 s^-1`. The earlier failure must not be used as a diagnosis of the actual Cook model.

The resting-threshold solve also had two independent errors: it omitted the permutation returned by `scipy.linalg.lu`, and it excluded chemical self-connections even though the neural current equation includes them. Both Python and Julia now use the same corrected solve and the audit checks its residual.

## Verified follow-up

The Cook workbook is required via `MODWORM_COOK_CONNECTOME` or an explicit path to `utils.construct_connectome_Cook`. With it, the Julia PPC runner completed zero input (4 samples), tiny input (4 samples), and small input (10 samples) with finite neural, muscle, and body trajectories. The optional stiff-body path now uses Sundials `CVODE_BDF` because the installed OrdinaryDiffEq 7 environment does not export `TRBDF2`; that path and the modulated Julia entry point also passed the 4-sample run. JuliaCall now activates the Conda environment's `julia_env` project when no explicit project override is set.

These are short-run smoke checks, not independent converged references. They do not activate the original PPC delayed feedback (starts at 1.18 s) or establish long-duration behavior. `python examples/diagnose_legacy_vode.py` remains available to assess legacy VODE with the correct Cook data; `--extra-vode` increases the internal VODE work limit only for diagnosis.

## Remaining required work for Milestone 2

1. Compare longer reference trajectories by refining the **coupling** interval, not just solver tolerances.
2. Compare legacy and shared-world coupling only when both complete, with their different PDF and muscle timing policies documented.
3. Validate longer trajectories and PPC onset after 1.18 s when computationally feasible.

Do **not** report a passing smoke test as proof of a high-accuracy reference or as repair of the legacy VODE integration path.
