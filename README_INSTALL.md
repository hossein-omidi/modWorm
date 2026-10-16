# modWorm Julia neuromodulation and population research package

**Target branch:** `dev-multiscale` of `hossein-omidi/modWorm`  
**Source inspected:** GitHub commit `52a3e83ecd5f25e744cd14062c569646b16f2768`  
**Status:** Focused Python contracts and short full-connectome Julia runs have been verified locally with Julia 1.13.1 and the Cook SI 5 workbook. This package is not a claim of verified speedup or of biologically validated collective behavior.

## Scope and safe installation

This repair updates existing model/Julia bridge code and notebooks, and adds the following files:

- `modWorm/julia_multiscale.py` — Python bridge/configuration, reproducible PDF analytic signaling, interface checks.
- `modWorm/_julia_modulated_ppc.jl` — small Julia RHS and Julia solver dispatcher, importing preexisting global Julia functions from modWorm.
- `notebooks/01_julia_baseline_and_equivalence.ipynb`
- `notebooks/02_serotonin_pdf_julia_experiments.ipynb`
- `notebooks/03_population_and_backend_boundary.ipynb`
- `notebooks/04_coupling_timescale_refinement.ipynb`
- `tests/test_julia_multiscale_contract.py`
- `docs/julia_multiscale_math.md`

Extract into the repository root, retaining paths. No overwrites of base files should be needed. **Do not replace the whole `modWorm` package directory**: merge the two new files into its existing directory. Check that the active git branch is the one you intend to modify.

### Runtime prerequisites

1. Working **original notebook** `1.4 Custom start (Full model).ipynb` using `juliacall` and a compatible Julia runtime.
2. Julia libraries already required by upstream `modWorm/__init__.py`: `DifferentialEquations`, `OrdinaryDiffEq`, `Sundials`, `SciMLBase`, `LinearAlgebra`, `LogExpFunctions`, `Interpolations`, etc.
3. Python `numpy`, `scipy`, `matplotlib`, `nbformat` or Jupyter, and the data files already in `modWorm/data`, `modWorm/muscle_maps` and `modWorm/presets_input`.

The Julia baseline and full-connectome notebooks also need the Cook SI 5 workbook. Download [the corrected Cook adjacency workbook](https://www.wormwiring.org/si/SI%205%20Connectome%20adjacency%20matrices,%20corrected%20July%202020.xlsx), then set `MODWORM_COOK_CONNECTOME` to its path before starting Jupyter, or pass the path to `utils.construct_connectome_Cook`. The checked-in `conn_gap_adjust_Cook.npy` and `conn_syn_adjust_Cook.npy` files are adjustment deltas and must not be used as complete connectomes.

To verify the Julia backend is available:

```powershell
python -c "from modWorm import require_julia; require_julia(); print('Julia ready')"
```

Julia's first-time package compilation/precompilation can be slow. **Do not** interpret that one-time startup as per-step integration speed. If the original custom-start Julia notebook cannot run, fix that environment before trying this package.

## Recommended execution order

```powershell
# Runs without Julia; usually under a second.
python -m unittest discover -s tests -p "test_julia_multiscale_contract.py" -v
```

Open `notebooks/01_julia_baseline_and_equivalence.ipynb`. Its first real simulation uses **4 time samples** (3 × 0.01 s). The notebook explicitly compares against original `proprioception_simulation.run_network_julia` with identical Cook connectome, muscle map, viscosity, initial conditions, and external stimulation. **Compare results before trusting further scenarios.**

Then open notebook 02 for MOD-1/PDF conditions, notebook 03 for independent population and optional Python contact, and notebook 04 for numerical coupling-interval sensitivity. Switch the flags `RUN_ALL`, `RUN_INTERACTING_PYTHON`, `MAKE_VIDEO` only after a short baseline succeeds.

## Why the integration is more efficient in principle

Original `shared_world.run_shared_world`: new Python SciPy BDF neural integration and new RK45 body integration **every coupling interval**, plus expensive Python array operations.

New `run_modulated_julia`: reuses Julia's `construct_NN_ppc`, `prep_network_integration_ppc`, `integrate_ode_ppc`, Julia body solver, Julia muscle mapping, and Julia neural current functions; initializes native CVODE_BDF/Tsit5 solvers **once per worm** and advances them across output intervals with `step!`. The optional stiff-body path uses CVODE_BDF because `TRBDF2` is no longer exported by the installed OrdinaryDiffEq 7 environment. Reuses original delayed PPC voltage-offset projection and original biomechanical equations. The new code is restricted to the extra MOD1/PDF RHS terms and a solver entry point.

No speedup has been measured in this execution environment. The model may still exhibit stiffness or ill-conditioning, and the Julia installation may need repair.

## Mathematical and biological scope

- MOD-1: current `g_i * A_5HT(t) * (V_i - E_Cl)` is **subtracted**, consistent with Python source.
- PDFR-1: occupancy `R_P(P)` and signal `tau_c dc/dt = alpha R_P - c` are updated analytically for a **piecewise-constant** concentration input. `I_PDF_i = gain*c_i` is added to neural RHS, with explicit **zero default gain**.
- For control callables, the new API evaluates concentration at output-grid boundaries and applies a **right-continuous zero-order hold** until the next boundary. This is a stated input-model difference from arbitrary continuous Python callbacks, not a hidden approximation. Place concentration jumps exactly on the grid and refine it as needed.
- Original proprioceptive projection `muscle_MapInv @ muscle_Map @ (V_delayed - Vth_delayed)` remains an offset to voltage, not a body-derived stretch current.
- Muscle activity is algebraic, with no invented calcium ODE.
- Original body-fluid interaction is its reduced drag model; **no Navier–Stokes PDE** is solved here.
- The existing Julia PPC runner uses sample-held muscle force, while body and neural solvers advance independently inside each interval. This is partitioned integration; it is **not** globally proven high order.
- There is no NSM neuron in the base 279-neuron map. Receptor targets, PDF current gain, and sensory-contact proxies are exploratory—not biologically calibrated.

## Population behavior: explicit distinction

- `run_independent_julia_population`: simulates independent complete worms with their own neural, body, MOD-1, and PDF objects; can analyze their geometrical spatial organization. **It does not couple their contact forces and cannot establish emergent swarming.**
- To model mechanical contact, notebook 03 calls the **existing Python** `shared_world.run_shared_world` with Python model classes and optional contact. This runs slowly in current upstream code and is intentionally off by default.
- A future accelerated, genuinely interacting Julia shared-world solver must move the coupled-body contact RHS and contact-to-neuron sensing together, preserving simultaneous body state and equal/opposite pair forces. We do not pretend that running separate Julia worms is equivalent.

## Numerical integrity and known limitations

**Threshold factorization:** The resting-voltage solve retains and applies the pivot matrix `P` returned by `scipy.linalg.lu` in both Python and Julia. It also includes chemical self-connections in the diagonal resting-current balance, matching the neural current equation. `audit_resting_threshold_factorization()` checks `P @ L @ U` against the original matrix and verifies the residual before Julia integration.

A short original Julia PPC baseline was run with 4 zero-input samples, 4 tiny-input samples, and 10 small-input samples; all returned neural, muscle, and body trajectories were finite. The same 4-sample baseline also passed with the optional stiff-body CVODE_BDF solver and with the modulated Julia entry point. These runs validate short-run execution only, not convergence or biological validity.

### What is verified in this package

- Python bytecode compiles.
- Notebook JSON/cells validate and compile.
- Eleven model-contract tests, nine Julia-contract tests, and the nine reference-contract tests pass (two expensive reference tests remain opt-in). The two opt-in full-connectome smoke tests also pass when the Cook workbook is configured.

### What still requires your machine

- Runtime improvement, convergence under interval refinement and response to neuromodulatory pulses.
- Several interacting worms (the Julia path is independent only).

If a Julia call errors, collect the **complete traceback and Julia error** without increasing tolerances or solver work caps as a workaround. We can then make an evidence-driven compatibility fix.
