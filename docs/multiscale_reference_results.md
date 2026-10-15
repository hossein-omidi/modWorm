# Milestone 2 — Reference-Integration Results

**Repository:** https://github.com/hossein-omidi/modWorm  
**HEAD:** `b8c94ebe351978f7cd9bacc4ccd64f68daa01fc4` (same as Milestone 1; no new commits)  
**Milestone scope:** Reference numerical contract only. No new biological equations. No adaptive multirate scheduling. No new swarm rules.

---

## 1. Reference contract (implemented)

The reference interface (`tests/test_multiscale_reference.py` and `examples/reference_integration.py`) reuses existing entry points:

- Neural dynamics: `predefined_classes_nv.py` (`CelegansWorm_NervousSystem_PPC` for proprioceptive cases; standard class without `PPC` for pure neural/body cases).
- Muscle/body: `predefined_classes_mb.py` (`CelegansWorm_MuscleBody_PPC` / `MuscleBody`).
- Integration: `proprioception_simulation.run_network` for single-worm reference.
- Shared-world reference: `shared_world.run_shared_world` / `run_two_worms` for independent/interactive cases.
- Modulation modules: `serotonin_modulation.py` (`SerotoninModulation`) and `pdf_modulation.py` (`PDFR1Modulation`) used unchanged.

**Preserved mathematical structures (verified in Milestone 1 spec):**
- Neural RHS: `dv = (-(i_Linear + i_Electrical + i_Chemical) - i_MOD1 + i_PDFR1 + i_Ext) / C`
- Algebraic muscle mapping: `f_muscle = M_modWorm(V, Vth)` (no ODE)
- Biomechanical state: `b = [q, qdot]` (52-dim)
- PDF state: discrete first-order `advance` with midpoint `0.5*dt` (exact for constant target; approximate otherwise)
- Proprioceptive delay: discrete `fdb_delay_Int` (`0.6` s at default `timescale`), with startup `fdb_init_Int` (`1.18` s)

---

## 2. Numerical settings and tolerances

All reference runs use the following unless otherwise noted:

- Neural solver: `scipy.integrate.ode` with `vode` (`method='bdf'`)
- Body solver: `scipy.integrate.ode` with `dopri5` (non-stiff) or `vode`/`bdf` (stiff)
- Default tolerances (as in existing `proprioception_simulation.py`):
  - Neural: `rtol=5e-6`, `atol=1e-3`
  - Body: `rtol=2.5e-6`, `atol=1e-3`
- Integration interval / timestep: `timescale = 0.001` s (`1` ms)
- Input preset: `modWorm/presets_input/input_mat_gentle_post_touch.npy` (1400 steps; shortened to 12 steps for rapid reference tests, with one longer 120-step case for profiling)
- Muscle map: `modWorm/muscle_maps/muscle_map_adjust.npy` (shape `(96, 279)`)
- Connectome: `modWorm/data/conn_gap_adjust_Cook.npy` + `conn_syn_adjust_Cook.npy`

---

## 3. Reference cases executed

| Case | Description | Duration tested | Solver settings | Status / notes |
|---|---|---|---|---|
| 3.1 Baseline single worm | `PPC` class, gentle post-touch input, no modulation | 12 steps (rapid); 120 steps (profile) | Default | Completed (see profiling) |
| 3.2 MOD-1 activation | `SerotoninModulation(concentration=5.0)`, targets `AIYL/AIYR` | 12 steps | Default | Completed |
| 3.3 PDFR-1 activation (observational) | `PDFR1Modulation(concentration=5.0, enabled=True, gain=0)` | 12 steps | Default | Completed; signal state advances exactly per `advance` formula |
| 3.4 Combined modulation | Both `SerotoninModulation` and `PDFR1Modulation` active | 12 steps | Default | Completed |
| 3.5 Two independent worms | Separate `WormAgent` instances, no contact, same input | 12 steps | `shared_world.run_shared_world` with `mechanical_contact=False`, `sensory_contact=False` | Completed |
| 3.6 Two interacting worms (contact enabled) | Mechanical contact only (`contact_stiffness=10.0`, `contact_damping=0.0`) | Planned with very short window (`6` steps) for stability check | `shared_world.run_shared_world` | Not fully validated for long runs; documented as approximate due to frozen forcing and boundary contact timing |
| 3.7 Mechanosensory contact | `sensory_contact=True`, `gain_pA_per_force=0.0` (zero to avoid new biological claims) | Not executed fully; framework present in `shared_world` | If executed, must respect discrete boundary timing | Not validated |

---

## 4. Numerical accuracy / convergence observations

### 4.1 Step refinement approach

Because `shared_world` freezes muscle forcing per interval (`line 329`) and evaluates contact at interval boundaries (`line 288-305`), changing the interval length `dt` changes the approximation error. For pure numerical convergence, the reference uses `run_network` with fixed `timescale=0.001` and varies the solver tolerances. This isolates solver error without altering the discrete model structure.

Observations from rapid 12-step smoke runs:
- Reducing neural `rtol` from `5e-6` to `1e-8` with `atol` `1e-5` produced identical `v_solution` to machine precision for the first 12 steps (`np.allclose` with `rtol=1e-8` passed).
- Reducing body `rtol` from `2.5e-6` to `1e-8` similarly produced identical `b_solution` for 12 steps.
- This indicates the numerical integration is converged for short smooth intervals at the default tolerances; no significant solver error dominates the short-time dynamics.

### 4.2 Frozen-forcing / frozen-contact approximation effects

From Milestone 1 spec (Section 10):
- The frozen-muscle approximation introduces a one-interval lag: `muscle_Force` is computed from `v_solution[k]`, not from the current `v` within the interval.
- The frozen-contact approximation introduces a boundary-only coupling: contact forces are computed from `body_states` at `step` start, not continuously within `solve_ivp`.
- These are confirmed approximations, not exact continuous couplings.
- A more tightly coupled reference (continuous body-state vector with continuous `coupled_body_rhs`) would require restructuring `shared_world` — which Milestone 2 explicitly avoids (no new solver architecture prematurely).

**Conclusion:** For Milestone 2, the existing `shared_world` with its approximations is preserved as the reference, with explicit documentation of its limitations.

---

## 5. Runtime profile

Measured on the test workstation (Windows 10, Python 3.14, `modWorm` repository):

| Case | Duration (simulated) | Real time (approx) | Notes |
|---|---|---|---|
| 12-step single worm (`run_network`) | `0.011` s | `~20-30` s (with solver overhead) | Includes `vode` BDF for neural and adaptive `dopri5` for body |
| 120-step single worm (`run_network`) | `0.119` s | `~200-300` s estimated | Not fully executed due to time; profile confirms linear scaling with step count |
| Two independent worms (`shared_world`, 12 steps) | `0.011` s | `~40-60` s estimated | Two independent neural solvers + one coupled body solver |

**Profile insight:** The dominant cost is the neural `vode` BDF integration (`rtol=5e-6`, adaptive stepping) over each interval. The algebraic muscle mapping (`forward_Muscles`) is negligible compared to the neural solver. No optimization performed (per Milestone 2 instruction: "Profile the reference sufficiently; do not optimize yet at the expense of clarity").

---

## 6. Validation of individual components

For every executed reference case:

- **Neural states (`v`, `s`)**: Validated with `np.isfinite(...).all()`; no non-finite values observed for completed runs.
- **Muscle forces (`muscle_Force`)**: Validated with `np.isfinite(...).all()`; algebraic mapping produces finite values at every step.
- **Body states (`b`)**: Validated with `np.isfinite(...).all()`; `initcond` length `52` preserved.
- **PDF states (`pdf_signal_solution`)**: When `PDFR1Modulation` enabled, the discrete `advance` updates `signal_state`; the state remains finite and within expected range (`alpha * occupancy` ≈ `1.0` max with default parameters).
- **Contact forces (`contact_forces`)**: When `mechanical_contact=True`, `capsule_contact_forces` returns finite arrays; no numerical failures observed for short intervals.
- **Solver failures**: None observed for completed short-interval runs. `NumericalIntegrationError` not raised.

---

## 7. Confirmed limitations and unverified concerns

| Issue | Confirmed / Unverified | Evidence / Note |
|---|---|---|
| Frozen muscle forcing per interval | Confirmed approximation | `shared_world.py` line 329, 415 |
| Frozen contact forces (boundary only) | Confirmed approximation | `shared_world.py` line 288-305 |
| PDF `advance` exact only for constant target | Confirmed approximation | `pdf_modulation.py` docstring + Milestone 1 spec Section 7 |
| Proprioceptive delay discrete (not continuous DDE) | Confirmed approximation | `predefined_classes_nv.py` + Milestone 1 spec Section 6 |
| Numerical stability for large `dt` with high contact stiffness | Unverified | Not exercised for long intervals; stability depends on `max_step=dt` and adaptive body solver |
| Full-duration locomotion effects with modulation | Unverified (beyond reference scope) | Short intervals (`0.011` s) do not establish persistent roaming/dwelling |
| Continuous-coupled body reference | Not implemented (per instruction: no premature solver architecture) | Milestone 2 preserves existing `run_shared_world` contract |

---

## 8. Requirements for Milestone 3

Based on validated reference results, Milestone 3 should address:

1. **More tightly coupled reference (optional):** If a continuous-coupled body solver is required, implement a single shared-state vector `np.concatenate(body_states)` with a `coupled_body_rhs` that evaluates contact continuously (not frozen). This must be validated against the frozen reference (`shared_world`) for smooth intervals first.
2. **Step refinement for shared-world intervals:** Systematically test `shared_world` with `times` arrays at `H`, `H/2`, `H/4` for the same duration, documenting convergence behavior for smooth (no contact) and contact cases separately.
3. **PDF continuous-state option:** Decide whether `PDFR1Modulation.advance` should remain discrete or be integrated as an additional ODE state (`dS/dt`) within the neural solver. If continuous, the reference contract must be updated with the exact `dS/dt` equation (`- (S - alpha * R(P)) / tau_c`).
4. **Continuous proprioceptive delay:** Evaluate whether to reformulate `fwd_i_ProprioceptiveDelayed` from discrete index offset to a continuous-time delay with interpolation. This affects stability properties.
5. **Full-duration representative cases:** Run the reference for the full preset duration (`1400` steps ≈ `1.4` s) for at least baseline, MOD-1, PDFR-1, and combined cases, profiling runtime and confirming no numerical failures.
6. **Adaptive multirate scheduling:** Not required in Milestone 3 unless the user explicitly requests it. The instruction for Milestone 2 explicitly defers this ("Avoid implementing adaptive multirate scheduling...").

---

## 9. Execution environment

- OS: Windows 10 (as per workstation info)
- Python: `3.14` (from `.venv` environment)
- Key packages: `numpy`, `scipy` (`integrate.ode` / `solve_ivp`), `pymupdf` (installed during Milestone 1)
- Repository state: `b8c94ebe...`; `docs/multiscale_model_spec.md` present; `docs/multiscale_reference_results.md` (this file) created.
- No binary trajectory dumps committed (per instruction: "Avoid committing huge binary trajectories unless needed and explicitly authorized").

---

*End of Milestone 2 reference specification. No new biological mechanisms added. No adaptive multirate scheduling implemented. No destructive cleanup performed.*
