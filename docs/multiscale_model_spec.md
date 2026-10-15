# modWorm multiscale model specification — Milestone 1

**Repository:** `hossein-omidi/modWorm`
**Inspected HEAD:** `b8c94ebe351978f7cd9bacc4ccd64f68daa01fc4`
**Previously reviewed SHA:** `b8c94ebe351978f7cd9bacc4ccd64f68daa01fc4` (identical; no newer commit was present).
**Scope:** Audit the implemented model and its numerical couplings. No biological equations, connectome data, or model code were changed.

This specification distinguishes implementation facts from assumptions and unresolved physical-unit questions. File references below are repository-relative and identify the primary source; line numbers are intentionally omitted because the source may evolve.

## 1. State vectors and dimensions

The repository contains several neuron/body class variants. The equations and dimensions below describe the Python base linear/chemical network and its PPC subclass, and the corresponding 24-segment muscle/body implementation. Other neuron classes can append channel-state variables; those variants are not silently folded into the base state definition.

| Component | State | Default dimension | Meaning |
|---|---|---:|---|
| Neural | `y_n = [v, s]` | `558 = 279 + 279` | Per-neuron voltage `v` and chemical synaptic activity `s`. |
| Body | `b = [q, qdot]` | `52 = 2(2 + 24)` | Generalized body coordinates and their velocities. |
| PDFR-1 signaling | `S` | 6 by default | One abstract signal per configured target; maintained outside the neural ODE by `PDFR1Modulation`. |

Here `q = [x_1, y_1, phi_1, ..., phi_24]`; `x_1,y_1` are the first centerline point and each `phi_i` is a segment angle. `qdot = [xdot_1, ydot_1, phidot_1, ..., phidot_24]`. The Python body integrator's full 52-state initial vector is initialized with these 26 coordinates followed by 26 velocities. Postprocessed centerline locations are derived quantities, not additional body states.

The standard neural initial condition contains two 279-entry blocks, generated with small random initial perturbations by `network_simulations.init_Initcond`. Channel-bearing variants append their channel states, so their total dimension is larger than 558. The chemical connectome and electrical connectome are passed into the network constructors and are not changed by this milestone.

**Primary evidence:** `modWorm/network_simulations.py`, `modWorm/predefined_classes_nv.py`, `modWorm/network_params.py`, `modWorm/body_simulations.py`, `modWorm/predefined_classes_mb.py`, and `modWorm/muscle_body_params.py`.

## 2. Implemented neural equations

For the standard Python base class, for neuron `i`, the RHS in `CelegansWorm_NervousSystem.forward_Network` is:

```text
C_i dv_i/dt = -I_L,i - I_gap,i - I_syn,i - I_MOD1,i + I_PDFR1,i + I_ext,i
ds_i/dt     = tau_rise,i (1 - s_i) sigmoid(B_i (v_i - vth_i))
               - tau_fall,i s_i
```

The current terms actually implemented are:

```text
I_L,i   = G_L,i (v_i - E_L,i)
I_gap,i = sum_j G_gap,ij (v_i - v_j)
I_syn,i = sum_j G_syn,ij s_j (v_i - E_rev,ij)
sigmoid(z) = 1 / (1 + exp(-z))
```

The constructors transpose the input connectivity maps into the model's internal indexing. The current formulas above follow `network_interactions.py`'s actual array operations; they are not a claim that a particular file's row/column naming convention is anatomical presynaptic/post-synaptic order. `G_gap` and `G_syn` are elementwise products of supplied connectome maps and the configured conductance maps, masked by the active-neuron mask. Preserve the supplied connectome matrices and their present orientation.

`vth` is an algebraic resting-potential calculation from `compute_vth_Linear`, not a state. The constructor precomputes an LU factorization for a linear system assembled from leak, gap, and estimated steady synaptic terms. At runtime `compute_vth_Linear(i_ext)` solves with right-hand side `bb_vth - i_ext`; thus external current affects both the voltage RHS and the activation threshold used in `ds/dt`.

### MOD-1 (serotonin)

`SerotoninModulation` implements a static Hill activation (no receptor kinetic state):

```text
A_5HT(t) = [5-HT](t)^h / (K_d^h + [5-HT](t)^h)
I_MOD1,i = g_MOD1 A_5HT(t) (v_i - E_Cl), for configured target neurons
I_MOD1,i = 0, otherwise
```

The MOD-1 term is **subtracted** in the neural numerator. The control can be constant or a time callable. Defaults are `g_MOD1=2 nS`, `K_d=1 µM`, `h=1`, and `E_Cl=-70 mV`; default targets are AIYL and AIYR. This is a chloride-conductance current, not a direct positive excitatory input.

### PDFR-1 (PDF-1)

`PDFR1Modulation` calculates Hill occupancy and evolves an abstract slow signal:

```text
O_PDF(t) = [PDF-1](t)^m / (K_P^m + [PDF-1](t)^m)
dS/dt    = (alpha O_PDF(t) - S) / tau_c
I_PDFR1,i = gain * S_i, for configured target neurons
```

The implementation's `advance(dt, time)` uses the exponential update described in Section 7. `I_PDFR1` is an optional phenomenological current-like bias added with a positive sign in the neural numerator; it is not an established PDFR-1 ion-channel law. Defaults are `K_P=1 µM`, `m=1`, `tau_c=5 s`, `alpha=1`, and gain `0 pA/signal-unit`; default targets are AIYL, AIYR, RIML, RIMR, RIAL, and RIAR. Therefore PDF signaling is observational with the default gain. The PDF-specific values are exploratory assumptions, as documented by `pdf_modulation.py` and `docs/research-extension.md`.

The Python neural RHS accepts a PDF signal explicitly and passes it to the hypothesis-current function. MOD-1/PDFR-1 are absent from some Julia-backed pathways; this specification does not imply otherwise.

**Primary evidence:** `modWorm/predefined_classes_nv.py`, `modWorm/network_dynamics.py`, `modWorm/network_interactions.py`, `modWorm/serotonin/serotonin_modulation.py`, and `modWorm/pdf_modulation.py`.

## 3. Muscle mapping is algebraic

The implemented muscle mapping is a pure algebraic function:

```text
f_muscle = M_modWorm(v, vth)
u         = muscle_Map @ (v - vth)
Ca_d      = max(u[0::4], 0) + max(u[1::4], 0)
Ca_v      = max(u[2::4], 0) + max(u[3::4], 0)
a         = (15 Ca)^2 / (1 + (15 Ca)^2)
f_muscle  = body_ForceScaling * a
```

`Ca` denotes the resulting dorsal/ventral algebraic activity inputs, not an independently integrated muscle-calcium state. There is no muscle ODE in `muscle_dynamics.py` or `predefined_classes_mb.py`; do not add one. `muscle_MapInv` is the pseudoinverse of the map and is used in the PPC projection, not as a muscle dynamic.

In the 24-segment implementation the force vector is split as `fR = force[:23]` and `fL = force[24:-1]` before computing curvature. The unassigned boundary entries are part of the current mapping layout; do not silently change slicing as part of a numerical rewrite.

**Primary evidence:** `modWorm/muscle_dynamics.py`, `modWorm/predefined_classes_mb.py`, `modWorm/muscle_body_params.py`.

## 4. Body RHS and force entry

The body state is `b = [q, qdot]`. Its RHS returns `[qdot, qddot]`; it does not integrate muscle state. From the current angles and generalized velocities, `forward_Body` derives segment centerline positions/velocities, local tangent/normal velocities, and body curvature. The implemented muscle-induced curvature for segment `i` is:

```text
k_i = 4 (fR_i - fL_i) w_i /
      (8 v_i w_i^2 - (fR_i + fL_i) h_i^2)
      * kappa_scaling + kappa_forcing
```

Here `v_i` is the body stiffness parameter, not neural voltage. This curvature enters the elastic/viscous bending moment; hydrodynamic normal/tangential forces then enter the generalized-coordinate acceleration equations. Specifically, `fwd_body_NormalTanForce`, `fwd_body_Forces`, `fwd_body_D2Angles`, and `fwd_body_D2Coordinates` compute the remaining RHS. The base body class uses its interpolated force history when no `muscle_force` argument is supplied; the PPC body class uses its current split muscle forces when no explicit force argument is supplied.

If a `(24, 2)` `contact_force` is provided, `forward_Body` adds its x/y nodal components divided by segment length to `Wx`/`Wy`, then recomputes the mass/length-weighted force differences consumed by the angular acceleration solve. The same augmented `Wx`/`Wy` contribute to translational acceleration. Thus contact enters the original body RHS as a mechanical force, not a neural state.

**Primary evidence:** `modWorm/predefined_classes_mb.py`, `modWorm/body_dynamics.py`, `modWorm/body_simulations.py`.

## 5. Original PPC feedback

The Python PPC implementation in `CelegansWorm_NervousSystem_PPC` is a delayed projected neural-voltage **offset**:

```text
if k >= fdb_init_Int:
    input_FDB = muscle_MapInv @ (muscle_Map @
                 (v[k - fdb_delay_Int] - vth[k - fdb_delay_Int]))
else:
    input_FDB = 0
v_for_currents = v_state + input_FDB
```

The offset is added to the voltage before leak, gap, chemical, MOD-1, and threshold-dependent activation calculations. It is not a current term divided by capacitance, and this method does not read body stretch, segment angle, curvature, or body velocity. The historical body/muscle arguments in the PPC method signature are not used by `fwd_i_ProprioceptiveDelayed`.

For this PPC class `timescale=0.01 s`, so `fdb_delay_Int=int(0.6/timescale)=60` samples (0.6 s) and `fdb_init_Int=int(1.18/timescale)=118` samples (1.18 s). Before history index 118 feedback is exactly zero; at index 118 the projection uses neural sample 58, which is 60 sample intervals earlier. These are discrete sample-index semantics, not a continuously interpolated delay.

Do not conflate this original PPC projection with the optional `shared_world.sensory_contact` proxy. That separate exploratory mechanism maps contact-force magnitudes at approximate body locations to touch-cell currents; it is disabled by default and is not the PPC feedback law.

**Primary evidence:** `modWorm/predefined_classes_nv.py`, `modWorm/network_interactions.py`, `modWorm/shared_world.py`.

## 6. Units and what is not specified

The included `data/documents/Celegans_model_units.pdf` gives the neuronal current-balance equation, capacitance/conductance/voltage/time units, resulting `dV/dt` units, external-current units, and a fluorescence conversion for synaptic activity. It does **not** specify biomechanical length, mass, force, fluid, muscle-map, or contact units. The source parameters have names and values but not a complete documented physical unit system for biomechanics.

| Quantity | Implemented value / unit | Evidence and qualification |
|---|---|---|
| Neuron count | 279 | `network_params.py`; `neurons.json` defines ordering. |
| `v`, `vth`, `E_L`, `E_rev`, `E_Cl` | mV | Model parameters and neuronal units PDF. |
| `C` | `0.0015 nF` = 1.5 pF per neuron | `network_params.py`. |
| `G_L` | `0.01 nS` = 10 pS per neuron | `network_params.py`. |
| Gap/synaptic conductance scale | `0.1 nS` maximum entries | `network_params.py`; actual entries depend on connectome maps. |
| External current | pA | Neuronal units PDF and neural RHS. |
| `s` | Dimensionless activity | Neural RHS; PDF describes fluorescent change relative to resting activity. |
| `tau_rise`, `tau_fall` | Stored as `1/1.5` and `5/1.5`; source labels seconds, but RHS uses them as coefficients multiplying `s` derivatives | `network_params.py` and `fwd_activity_Chemical`. Do not treat these values directly as simple rise/fall time constants without deriving the resulting rate. |
| `B` | `0.125 mV^-1` | `network_params.py`. |
| PPC/network class `timescale` | `0.01 s` | `CelegansWorm_NervousSystem_PPC`, `_sg`, and corresponding Julia constructors. `network_params.CE.timescale` separately says `0.001 s` but is not used to set those constructors' `self.timescale`. |
| Body segment count / lengths | 24; `h` entries 1 for first 20 and 0.1 for last 4 (unit unspecified) | `muscle_body_params.py`. |
| Body parameter values | `E=600000`, radii `a=b=0.325`, `rho=2`, `rho_f=2`, `mu=0.01`, `C_N=2`, force scale 6000 | `muscle_body_params.py`; the source does not provide a complete set of physical units or a dimensional audit. |
| MOD-1 concentration, affinity | µM; default 0 and 1 | `serotonin_modulation.py`. |
| PDF-1 concentration, affinity | µM; default 0 and 1 | `pdf_modulation.py`. |
| PDF signal `S` | Dimensionless normalized cAMP-like state per module docstring | Exploratory scale; not experimentally calibrated. |
| PDF `tau_c` | 5 s default | `pdf_modulation.py`; exploratory assumption, not fitted. |
| PDF gain | `0 pA/signal-unit` default | `pdf_modulation.py`; zero means no default neural effect. |
| Contact stiffness/damping and sensory gain | Defaults 10, 0, and 0 respectively; physical units as declared or implied by API are not experimentally calibrated | `collective.py`, `shared_world.py`; geometry/unit consistency remains unresolved. |

The neuronal combination nS·mV gives pA numerically in the model's stated unit convention, and pA/nF gives mV/s. The included PDF prints its own current-balance formula; the audited RHS above follows the executable Python source. No extra global time divisor appears in that Python RHS.

## 7. PDFR-1 signal update: exactness and approximation

The source update is:

```text
target = alpha * occupancy(time + dt/2)
S_new  = target + (S_old - target) exp(-dt/tau_c)
```

This is the exact solution over one interval for `dS/dt = (target - S)/tau_c` **if target is constant during the interval**. For a time-varying concentration, the midpoint occupancy is frozen over the interval: this is an approximation to the nonautonomous ODE. A midpoint quadrature for the forcing gives a second-order local update for smooth forcing when considered as an approximation to the exact exponential convolution; the method does not become exact merely because the midpoint is used.

Scheduling differs by driver:

* `proprioception_simulation.integrate_ode` calls `advance(timescale, r_nv.t)` once before each neural interval. The returned new `signal_state` is then read by the neural RHS throughout that interval. The RHS does not continuously integrate `S`.
* `shared_world.run_shared_world` calls `advance(dt/2, time_start)` before the neural solve, so this half-step samples occupancy at `time_start + dt/4`. It then calls `advance(dt/2, time_start + dt/2)` after the neural solve, sampling at `time_start + 3dt/4`. Together these are two half-step exponential updates and advance the stored signal by the full `dt`; the neural solve uses the intermediate half-step signal held constant over its interval. For constant target, the two updates compose to the exact full-interval update. For varying target, they are a two-midpoint approximation.

The `S` state has six entries by default because the configured target list has six neurons; it is an external module state, not part of the returned neural ODE vector unless a future implementation explicitly changes that contract.

## 8. Numerical timescales derived from source

| Process | Value / derivation | Interpretation |
|---|---|---|
| Isolated passive neural voltage | `C/G_L = 0.0015/0.01 = 0.15 s` | Leak-only time constant; gap, synapse, neuromodulation, and input change the effective dynamics. |
| Neural output sample interval | `0.01 s` for Python PPC and `_sg` classes | The Python `network_params.CE.timescale=0.001 s` is a separate parameter and does not override the constructors. |
| Chemical activity dynamics | RHS coefficients `tau_rise=0.667` and `tau_fall=3.333` | These appear as rate coefficients (per second dimensionally), not as a standard exponential rise/fall kernel. At fixed voltage, local relaxation rate is `tau_rise*sigmoid(B(v-vth)) + tau_fall`; its time constant is the reciprocal of that sum. The source's units label deserves clarification. |
| PPC delay | 60 samples × 0.01 s = 0.6 s | Discrete delay index for PPC classes. |
| PPC startup | 118 samples × 0.01 s = 1.18 s | Feedback remains zero until the index threshold. |
| PDF signaling | `tau_c=5 s` default | Abstract exploratory signal decay constant. |
| Body/contact integration in shared world | RK45 per sample interval; `max_step=dt` | Adaptive internal steps are permitted, bounded above by the user-supplied sample interval. Actual counts depend on the trajectory and tolerances. |
| Neural integration in shared world | BDF restarted over each sample interval | Each interval receives the prior endpoint as its initial state; per-interval tolerances are user-configurable. |
| Muscle recomputation | Once per sample interval before body solve | Force held fixed during that interval's body integration. |

No biological rate is inferred from solver tolerances or invented to fill a documentation gap.

## 9. Dependency map

```text
Imposed/time-controlled inputs
  input_mat or external_current(t) ───────────────┐
  serotonin concentration(t) ── Hill activation ──┤
  PDF-1 concentration(t) ── Hill occupancy ───────┤
                                                   v
ODE states                                    neural RHS
  v (279), s (279) ───────────────────────────────┤
  q, qdot (body: 52 total) ── body RHS ───────────┤
  S_PDF (6 default; updated outside neural ODE) ──┘
                                                   |
Algebraic quantities                              |
  vth = linear solve(i_ext) ──────────────────────┤
  muscle force = M_modWorm(v, vth) ───────────────┤
  curvature, drag and generalized body forces ────┤
  capsule contact forces = geometry(body states) ─┘

Delayed/history quantity (PPC only)
  v[k - 60] - vth[k - 60]
       └─ muscle_MapInv @ (muscle_Map @ ·)
            └─ input_FDB voltage offset (zero for k < 118)
```

The optional shared-world sensory-contact proxy is another boundary-sampled imposed neural current derived from contact force. It is separate from the historical PPC projection. In shared-world runs, the mechanical contact force itself is recomputed inside body RHS calls.

## 10. Existing integration/coupling schedules

### `proprioception_simulation.run_network`

* Builds equally spaced output times from `NN_nv.timescale`, interpolates external current, and advances independent legacy `scipy.integrate.ode` solvers: neural VODE/BDF and body DOPRI5 (or VODE/BDF when `stiffness_mb=True`).
* At each sample index `k`, computes `vth[k]`, evaluates muscle force from stored `v[k]`, computes PPC from stored histories, optionally advances PDF by a full step, then integrates neural and body states separately to the next sample.
* The body RHS uses the muscle force stored in `NN_mb.fR/fL` for PPC body class. That force is therefore frozen from the interval-start neural sample while body evolves. Neural and body are not jointly integrated and no same-interval feedback of new body motion into neural activity is provided by this schedule.
* PPC becomes available only when the discrete startup threshold is reached; its source sample remains the specified discrete delay.

### `shared_world.run_shared_world`

For each user-supplied `[time_start, time_stop]`:

1. Sample pairwise capsule contacts at the current body states. Optionally convert those forces to touch-cell current at approximate sensor positions; this sensory current is held over the neural interval.
2. Evaluate each agent's `vth` and algebraic muscle force from the current neural voltage and boundary external-plus-sensory input. Compute PPC from stored sample histories where the network provides it.
3. Advance PDFR-1 by a half-step, then integrate each agent's neural state separately using `solve_ivp(BDF)` over the full interval. The external current interpolator is evaluated at neural RHS times; boundary contact sensory current and the intermediate PDF signal are fixed through that solve. BDF is restarted per interval.
4. Concatenate all body states and integrate them with one `solve_ivp(RK45)` over the same interval, `max_step=dt`. Mechanical capsule contact is recomputed from the trial body states on each body RHS evaluation; muscle forces remain the copies computed at step 2. Body integration is therefore mechanically coupled across worms, but not jointly integrated with their neural systems.
5. Advance PDFR-1 by the second half-step and save endpoint histories. Histories sampled for neural/PPC outputs are at the shared sample boundaries.

This is an operator-split, boundary-coupled multirate schedule—not a monolithic integration of neural, signaling, body, and contact states. The restart cost and absence of cross-interval error control for the coupled problem follow from this design. Contact damping uses body velocities generated by the current body trial state; mechanosensory neural current does not use those trial-time contacts.

## 11. Consequential limitations and evidence classification

“Confirmed” here means the behavior follows directly from the current implementation; “defect” is reserved for a demonstrated contradiction with the stated intended contract, not merely a modeling or accuracy tradeoff.

| Classification | Finding | Consequence / evidence |
|---|---|---|
| Confirmed approximation | `shared_world` freezes the muscle-force vector for the entire body sample interval. | It does not feed within-interval neural evolution back into body forcing. See boundary evaluation before neural solve and the copied force passed to `coupled_body_rhs`. |
| Confirmed approximation | `shared_world` samples touch-cell sensory current at the interval's initial body state. | Contact changes during body integration are not reflected in neural input until a later boundary sample. The time lag is bounded by the sample interval, not a fixed extra interval in all trajectories. |
| Confirmed architecture | Neural solvers are independently restarted per interval; body solvers are restarted per interval, with one concatenated body solve across agents. | Neural/body coupling is operator-split; body-body mechanical contact is evaluated during body integration. There is no shared coupled neural/body solver or global error estimate. |
| Confirmed architecture | PDF signal is advanced externally rather than included in neural ODE state. | Neural dynamics see a constant signal during each interval; update accuracy depends on interval size and concentration variation. Two half-steps in shared-world use quarter/three-quarter forcing samples. |
| Confirmed architecture | Original PPC delay uses sample indexing plus a separate startup threshold. | It is a discrete delayed voltage offset, not continuous-time interpolation, and not body-derived stretch feedback. |
| Confirmed modeling limitation | Contact-to-neuron conversion is a separate configurable proxy, with approximate sensor locations and uncalibrated current gain; disabled by default. | It is not evidence of calibrated mechanosensation or of the original PPC equation. `shared_world.py` and `docs/research-extension.md` label this exploratory. |
| Confirmed documentation mismatch | `network_params.CE.timescale` is 0.001 s, but PPC and `_sg` Python constructors set `self.timescale=0.01 s`. | Specs or tests must use the instantiated class's timestep, not assume the params field controls it. |
| Confirmed unit gap | The supplied units PDF specifies neuronal but not biomechanical/contact units. | Dimensional consistency of body, muscle, and contact parameters cannot be established from the repository documentation alone. |
| Unverified concern | Whether the model's shared-world results converge as the sample interval shrinks, particularly under stiff/high-frequency contact or changing stimuli. | Requires a step-refinement study; no convergence claim is made here. |
| Unverified concern | Whether an interpolated continuous-time PPC delay is biologically intended or whether the existing discrete delay is the required model contract. | The code confirms only the discrete behavior; biological suitability requires source-paper/research verification. |
| Unverified concern | Whether the muscle/body parameter values and contact-force adapter are dimensionally consistent with a physical unit system and experimentally calibrated. | Documentation lacks the required biomechanical unit declarations and calibration evidence. |
| Unverified concern | Whether the synaptic `tau_rise`/`tau_fall` names and seconds comments intend direct rate coefficients as implemented. | The RHS and numeric values are confirmed; the semantic label discrepancy is not resolved by the available source. |
| Not a defect on present evidence | The two shared-world PDF half-steps. | They sample at quarter and three-quarter interval times and compose to an exact full step for constant target. For varying target they are an approximation; calling the conventions inconsistent is not supported by the code. |
| Not a defect on present evidence | `max_step=dt` for the body solve. | This only limits internal step length; adequacy depends on trajectory, tolerances, and contact stiffness and remains unverified. |

## 12. Conservative reference-integration contract for Milestone 2

This is a proposed numerical contract, not an implementation in this milestone:

1. Preserve the current neural RHS, parameter arrays, connectome matrices, MOD-1/PDFR-1 signs and target configuration, algebraic muscle map, and body RHS. Record baseline trajectories and solver inputs before any solver refactor.
2. Keep muscle activity/force algebraic; do not introduce a muscle ODE. Keep the biomechanical ODE state `b=[q,qdot]` and the existing force-entry conventions.
3. Define a single explicit synchronization interval and time grid; make external inputs and receptor controls evaluable at solver time. Report convergence under interval refinement rather than claiming the sample interval is inherently adequate.
4. At each coupled RHS evaluation, use the current trial neural/body states for any quantities the chosen reference contract declares coupled. Keep mechanical contact geometry and its equal-and-opposite force accumulation explicit. Treat contact-derived neural sensory input as a separately configured hypothesis, never as a replacement for original PPC.
5. Preserve PPC's current discrete sample-delay/startup semantics in the compatibility reference, with history buffers initialized and indexed explicitly. If a continuous-delay alternative is later proposed, label and validate it as a distinct numerical/modeling contract.
6. Integrate the PDF signal either with its stated first-order equation and controlled solver state, or retain the exponential update with explicit forcing sample times; verify both against the constant-target exact solution and a time-varying reference. Do not change the PDF biological hypothesis or default zero gain.
7. Require state dimension, finite values, solver success, contact force balance, and time-grid convergence checks. Compare the refactored zero-contact/zero-neuromodulation trajectories with the preserved implementation before enabling new coupling semantics.

## 13. Concise Milestone 2 plan

1. Create focused regression tests for actual default dimensions, the isolated leak timescale, algebraic muscle output, PPC delay/startup/projection, PDF constant-target exactness, and the coupled body/contact RHS.
2. Implement the reference integrator around the contract above without altering biological equations; make interval and tolerances explicit and keep the current sample-based mode available for comparison.
3. Add interval-refinement and legacy-versus-reference comparisons for single-worm and multi-worm/contact cases; report runtime separately from correctness and convergence.
4. Document any intentional scheduling differences and remaining unit/calibration assumptions before proceeding to later research milestones.

## 14. Evidence and validation performed

* Recorded the current HEAD and verified it exactly matches the previously reviewed SHA.
* Inspected the requested neural, interaction, neural-class, muscle, body, PPC, shared-world, modulation, collective-contact, and parameter source files.
* Read the two-page `data/documents/Celegans_model_units.pdf` using `pdfinfo`/`pdftotext`. MiKTeX emitted an environment compatibility warning but returned the PDF's extracted text.
* Cross-checked the README research description/citation and `docs/research-extension.md`, including its cautions about uncalibrated PDFR-1/contact hypotheses and absent roaming/dwelling validation.
* `python -m unittest tests.test_research_extensions tests.test_serotonin_modulation`: **40 tests passed** in 275.571 s.
* An attempted run including the pre-existing `tests.test_multiscale_reference` emitted VODE “Excess work done” warnings and repeated test errors without completing; that run was stopped and is not counted as passing evidence. The focused 40-test run above completed successfully.
* No model code or test source was modified as part of this milestone. Test execution updated tracked/untracked Python bytecode caches; those existing/generated worktree artifacts were left untouched.

*End of Milestone 1. No Milestone 2 implementation performed.*
