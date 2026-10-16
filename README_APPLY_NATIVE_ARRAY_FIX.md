# modWorm Julia native-array fix — 2026-10-15

## Root cause

Notebook 01 failed at the original `sim.run_network_julia` call, *before*
ODE integration, because JuliaCall passed NumPy array fields into Julia as
`PythonCall.PyArray`. Sundials constructs native Julia `Vector{Float64}`
solution snapshots, which cannot be appended to a `Vector{PyArray{Float64,1,...}}`.

## Fix

For the legacy Python entry points `run_network_julia` and
`run_network_julia_ensemble`, call `Main.modworm_native_ppc_dict` on the model
parameter mappings **before** passing them to `Main.run_network_ppc`.
`modworm_native_ppc_dict` copies each NumPy-backed array into a native Julia
`Array` only once at the Python->Julia boundary. It leaves scalars alone.

The replacement `modWorm/_julia_modulated_ppc.jl` uses the *same conversion* for
the newly added serotonin/PDF path before constructing either `ODEProblem`.
No biological equation, original muscle mapping, receptor law, body force law,
solver tolerances, or model initial value is deliberately changed.

## Apply (from your repository root in PowerShell)

1. Shut down the running notebook and **restart the Jupyter kernel**, especially
   after Julia/SciML package precompilation indicated loaded-version mismatches.
2. Extract this ZIP at the repository root. **Merge folders** rather than
   replacing the whole `modWorm` directory.
3. Preserve your current git status or take a backup/commit of existing edits.
4. Run:

```powershell
python apply_julia_native_arrays.py
python -m unittest discover -s tests -p 'test_julia_native_bridge_patch.py' -v
python examples/verify_julia_native_bridge.py
```

The installer makes a backup named
`modWorm/proprioception_simulation.py.before_julia_arrays.bak` and refuses
unexpected source versions or ambiguous entry points. It is idempotent.

5. Restart the Jupyter kernel again if it was already open before installation.
6. Run notebook 01. For a low-cost first numerical attempt, use only two
   time samples (`INPUTS = INPUTS[:2].copy()`), then restore the intended
   input length once the original and modulated pathways both execute.

## What this does not establish

- It fixes the array-type mismatch; it does not prove that the full Julia ODE
  simulation is numerically stable, equivalent to the legacy equations, or fast.
- Startup precompilation time must be excluded from steady-state benchmarks.
- The Python multiworm contact solver is *not* made Julia-native by this patch.
- The original Julia PPC implementation has other possible performance and
  calibration limitations not addressed here.

## File contents

- `apply_julia_native_arrays.py`: safe source transformer for the legacy entry points.
- `modWorm/_julia_modulated_ppc.jl`: replacement for the previously delivered accelerator module.
- `examples/verify_julia_native_bridge.py`: native Julia type/value smoke check.
- `tests/test_julia_native_bridge_patch.py`: fast patch regression checks.

The original `modWorm/julia_multiscale.py` is unchanged and remains compatible.
