# Milestone 1 review and targeted corrections

**Target:** `hossein-omidi/modWorm`, branch `dev-multiscale`, inspected commit `52a3e83ecd5f25e744cd14062c569646b16f2768` (2026-10-15 review).  
**Scope:** Source-backed equation/clock/contract review. **Not** a new numerical integrator, a biological calibration, or a claim of validated collective behavior.

## 1. Decision: preserve the scientific core

The existing `docs/multiscale_model_spec.md` is substantially useful and should remain in place. The original 279-neuron model, chemical/electrical interaction matrices, algebraic motor-to-muscle mappings, 24-segment viscoelastic body RHS, original discrete PPC feedback, MOD-1 current, PDFR-1 first-order reduced state, and generalized `shared_world` are retained **unchanged** by this patch. There is no demonstrated Milestone-1 reason to replace their governing equations.

**New targeted verification:** `tests/test_multiscale_model_contract.py` checks selected receptor laws and the original feedback/muscle calculation directly. It does not claim numerical convergence, physical calibration, or biological validation.

## 2. Confirmed issue in the already-present Milestone-2 tests

In the previously committed `tests/test_multiscale_reference.py`, `test_two_independent_worms` constructed PPC neural objects with `network.timescale = 0.01 s` (from `CelegansWorm_NervousSystem_PPC`), then supplied a shared-world grid of `0.001 s` per step:

```python
# Original, inconsistent with the instantiated PPC model
# times = np.linspace(0, (len(input_short)-1)*0.001, len(input_short))

# Correct: use the actual instantiated model timestep
times = np.arange(len(inputs), dtype=float) * nn_a.timescale
```

For the **original 12-sample test**, the old grid represented **0.011 s** whereas the PPC class clock implied **0.11 s**. The corrected smoke test now intentionally uses **two samples spanning 0.01 s** to keep its computational cost low. The former mismatch invalidated the physical-time interpretation of the test; both short versions end before PPC feedback startup.

The replacement test file also removes the synthetic/random muscle-map fallback (which would not be a canonical reference), uses the real `(96,279)` map and complete canonical `(steps,279)` input rather than broadcasting neuron 0's input across all neurons, checks for the PDF history instead of accepting its absence, and seeds model initialization for repeatability. These changes concern **tests only**, not model equations.

**Important:** These are *smoke/regression* tests. Finite outputs for 12 points are not evidence of a converged or high-accuracy reference. The current original PPC starts at `fdb_init_Int=118` on 0.01-second samples, so the 12-sample tests cannot validate active delayed feedback.

## 3. Current modeling contracts, source-grounded

| System | Verified current implementation | Important qualification |
|---|---|---|
| Neural state | In basic Python model, `y = [v(279),s(279)]`, 558 real states. | Some other neuron classes add channel states. |
| Neural RHS | `C dv/dt = -(I_linear+I_electrical+I_chemical) - I_MOD1 + I_PDFR1 + I_external` (with contact current added to external input where enabled). | The PPC pathway uses a delayed **voltage offset** applied inside `forward_Network`, not an injected current. |
| MOD-1 | `I_MOD1 = g * A_5HT(t) * (V-E_Cl)`, `A_5HT = S^h/(Kd^h+S^h)` on configured neurons. | Default `g=2 nS` is exploratory, much larger than the standalone `0.01 nS` leak and should not be presented as fitted physiology. At 5 µM with Kd=1, activation is 5/6: effective g ≈ 1.667 nS, ~167 times the isolated leak conductance for a target cell. |
| PDFR-1 | `dc/dt=(alpha*R_P(P(t))-c)/tau_c`; downstream bias is optional and by default zero. | The two half-step exponential updates are exact for a constant target; approximate under varying concentration. This is a *reduced hypothesis* rather than a verified cell-specific GPCR current. |
| Muscle | Algebraic `muscle_Map @ (v-vth)` → positive rectification/pooling → saturating activation → scaled force. | There is **no independent muscle ODE**; do not introduce one under the guise of a numerical refactor. |
| Body | `b=[q,qdot]` with 24 angles and two translated reference coordinates; 52 ODE states. | Segment radii/length parameters exist but the full physical unit/calibration system, particularly for the new contact adapter, remains unresolved. |
| PPC | At sample k≥118, apply `muscle_MapInv @ muscle_Map @ (v[k-60]-vth[k-60])` as a voltage offset. | 0.6 s nominal delay and 1.18 s startup apply only on the original **0.01-second PPC sample clock**; this is not direct body stretch. |
| Shared world | Independent neuronal integrations and a single concatenated, contact-coupled body RHS. | Muscle forcing and contact-derived sensory currents are held at sample boundaries. Coupled convergence and long-duration accuracy have not been demonstrated. |
| Population | Supports any positive number of agents programmatically. | Generality of API does not imply practical full-connectome multiworm scalability or emergent swarming. |

Relevant source files: `modWorm/predefined_classes_nv.py`, `modWorm/network_dynamics.py`, `modWorm/network_interactions.py`, `modWorm/muscle_dynamics.py`, `modWorm/predefined_classes_mb.py`, `modWorm/muscle_body_params.py`, `modWorm/shared_world.py`, `modWorm/serotonin/serotonin_modulation.py`, `modWorm/pdf_modulation.py`.

## 4. Mathematical points that remain open, not fixed by assertion

1. **Reference accuracy:** The existing Milestone-2 test suite does not establish a converged reference using interval refinement. Its documentation reports preliminary/estimated runtimes and some uncompleted scenarios. Such claims must not be treated as independently verified solver results. Milestone 2 must re-run and correct the reference report.
2. **Time alignment:** Original PPC uses discrete sample indices; an adaptive shared-world physical-time delay would require a separate, explicitly validated history implementation. This patch does **not** change the existing PPC model.
3. **Body/contact force semantics:** Newton-pair balance at the capsule level is not alone a proof that the force-density mapping satisfies generalized virtual work and torque consistency. The existing physical-unit gap remains a major next-stage audit item.
4. **Contact and motor coupling:** The current shared-world scheme freezes motor forces and sensory-contact currents over macro intervals. That is an approximation, not automatically a software defect. Its error must be measured before replacement.
5. **PDF time forcing:** The constant-occupancy exponential law has an analytical solution. Arbitrarily varying PDF concentration instead requires controlled integration of the exponential convolution, not an unsupported claim of exactness.
6. **Biological scope:** Externally controlled receptor activation is not endogenous serotonin/PDF state regulation. No persistent roaming/dwelling, aggregation, or CASY-1-driven swarming is demonstrated by these short numerical tests.

## 5. Run tests on the actual repository

From the repository root with the original project dependencies installed:

```bash
python -m unittest discover -s tests -p 'test_multiscale_model_contract.py' -v
python -m unittest discover -s tests -p 'test_multiscale_reference.py' -v
```

The first command is a **fast unit/contract check**. The second exercises full numerical code and can be expensive; it is a Milestone-2 smoke suite rather than a verified reference. Individual numerical tests may encounter VODE or body solver failure, which must be reported, not silenced.

## 6. Patch verification and boundaries

The ZIP contains only `docs/multiscale_milestone1_review.md` and two tests: a **new** mathematical-contract test and a **replacement** of the previously present Milestone-2 smoke-test file to remove verified time/input/data problems. There are **no changes** to `modWorm/*.py`, datasets, notebooks, user-generated videos, or original tutorial code.

**Checks performed in this delivery environment:** Python parsing/bytecode compilation of the packaged tests, archive integrity, and dependency-free source-reproduction calculations where applicable. The GitHub-connected branch was read for the mathematical audit. The complete remote repository could not be checked out in the local execution sandbox due restricted external network access, so end-to-end model tests have **not** been independently executed here. Do not claim otherwise.

**Next milestone:** Establish a demonstrably converged high-accuracy numerical reference for representative original-model cases; keep the existing models unmodified unless a source-backed defect is confirmed.
