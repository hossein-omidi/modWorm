# Mathematical contract: Julia MOD-1/PDF-1 and original PPC body solver

## Baseline dynamics

For `n=279`, `y_N=[v,s]∈R^(2n)` and body state `y_B=[q,qdot]∈R^52`. Dynamics are inherited from modWorm; no reduction or replacement of the connectome:

```
C_i dv_i/dt = -I_L,i(v) - I_gap,i(v) - I_syn,i(v,s)
              - I_MOD1,i(v,t) + I_PDFR1,i(t) + I_ext,i(t)
ds_i/dt = a_i(1-s_i) logistic(B_i(v_i + u_PPC,i - vth_i)) - b_i s_i
```

All ionic and gap/chemical currents operate on `v_eff=v+u_PPC`, exactly as in the upstream Julia PPC class, while the state variable remains `v`.

`vth` is the same algebraic threshold computed by `compute_Vth`; the original `muscle_Map` and `forward_Muscles` produce force algebraically from sampled `v-vth` at each boundary. `forward_Body!` uses the existing 52-state rod/fluid drag equations, with that force held over each sample interval.

## MOD-1

```
A_5HT(t) = [5HT]^h / (K_d^h + [5HT]^h)
I_MOD1,i = g_MOD1,i * A_5HT(t) * (v_eff,i - E_Cl)
```

`I_MOD1` enters with a negative sign. The activation is evaluated from `SerotoninModulation.receptor_activation(t_k)` at each boundary and held over `[t_k,t_(k+1))`. Target conductance is zero for non-target neurons. No receptor kinetics are invented.

## PDF-1/PDFR-1

```
R_P(P) = P^m / (K_P^m + P^m)
tau_c * dc_j/dt = alpha * R_P(P(t)) - c_j
I_PDFR1,i(t) = g_bias * c_j(t),   for assigned target j→i
```

Concentration is externally prescribed, **not** a released-neuropeptide spatial PDE or reconstructed network-derived concentration. Within interval `k`, `P(t)=P_k`; define `Q_k=alpha R_P(P_k)`:

```
c_j(t) = Q_k + (c_j(t_k)-Q_k)exp(-(t-t_k)/tau_c), t_k≤t≤t_(k+1)
c_j(t_(k+1)) = Q_k + (c_j(t_k)-Q_k)exp(-H_k/tau_c)
```

This is the exact solution of the first-order signaling ODE for the **piecewise-constant external control law**. The implementation computes its boundary knots once in Python and evaluates it from Julia **at each native neural RHS time**. This is more temporally consistent than holding a single intermediate signal through the full neural interval, but it changes the numerical coupling semantics from `shared_world.py`; direct numerical comparisons must account for that difference.

Default `g_bias=0`; any nonzero gain is a phenomenological neural bias rather than an established receptor-specific ionic current.

## Delayed PPC

For output-grid index `k`, the existing model applies

```
u_PPC(k) = muscle_MapInv @ muscle_Map @ (v[k-D] - vth[k-D]), if k≥K0
          = 0, otherwise
```

with `D=int(0.6/timescale)` and `K0=int(1.18/timescale)` at default values. It is a **sampled voltage offset**, not a biological body-stretch observation. The new Julia implementation calls original `integrate_ode_ppc` and original `proprioceptive_FDB`, preserving their sample semantics. Refined `timescale` changes `D` and `K0` and is valid only if physical delays remain matched.

## Multiscale interpretation

`tau_leak=C/G_L≈0.15 s` under isolated leak and default `tau_PDF=5 s`, giving `epsilon≈0.03`. This separation is only suggestive: network eigenmodes and contact may be much faster. The present solver uses CVODE's adaptive **internal** neural steps and analytical PDF signal updates; it does **not** introduce an unvalidated adaptive multirate macro-step scheduler. `H` is the synchronization/sample interval for original PPC and muscle forcing. Notebook 04 compares different `H` with the same physical duration.

## Mathematical conditions for a numerical reference

- Identical initial state, original connectome, muscle map, viscosity and imposed stimulus.
- Full state dimensions, finite outputs, and Julia solver successful endpoints.
- Validation of original resting-threshold LU matrix including potential pivot permutation.
- Reference comparison original Julia vs new zero-mod Julia.
- H/H2/H4 refinement and perturbation tests. Compare time-aligned trajectories, not trajectories at different times.
- Contact-enabled collective validation remains separate in the upstream Python shared-world code. No Julia multiworm physical interaction is claimed.

## Known model limits

The 279-neuron network has no NSM in its data map; default PDF current gain zero; the biomechanical environment is the existing drag law rather than a resolved PDE. These limitations affect biological interpretation but do not justify inserting hidden locomotor/speed controllers.
