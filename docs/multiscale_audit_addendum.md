# Addendum to Milestone 1 model audit — Milestone 2 integration review

The pre-existing `docs/multiscale_model_spec.md` describes the audited
model's equations and original `b8c94ebe` snapshot. The target branch for
this update is `dev-multiscale` at `52a3e83`. The historical SHA inside the
Milestone 1 specification must not be confused with the branch's later HEAD.

## Confirmed pipeline dependencies

- The neural RHS resides in the original neuronal classes and calls the
  existing network dynamics and interaction functions. This patch does not
  create a duplicate neuronal ODE.
- The muscle pipeline computes algebraic outputs from current voltage and
  threshold through the existing muscle map; no new calcium state exists.
- Body integration calls the original biomechanical RHS; no new rod or
  particle dynamics are introduced.
- `shared_world` evaluates contact during the RK45 body RHS, but samples the
  touch-cell proxy at neural interval boundaries. The muscle input is also
  held between coupling boundaries. The description 'frozen contact forces'
  is **incorrect** if applied to the body RK45 integration.
- PDF uses the existing first-order signal updater. With its default zero
  gain, it has no effect on the neuronal RHS. That is an observational control,
  not evidence for PDF-dependent locomotion.
- PPC's original neural projection is a *voltage offset* with discrete
  sample delay. In this milestone, it is not reinterpreted as a current or
  converted to a DDE.
- The `proprioception_simulation` legacy VODE route and `shared_world` route
  are distinct methods with distinct limitations. They must not be described
  as numerically equivalent without executing comparisons.

## Why this update deliberately makes no biological-model source changes

The observed VODE 'excess work' warning is a numerical failure, but its root
cause has not been proven. Changing neural conductances, receptor values,
original feedback delays, body stiffness or contact forces to suppress the
warning would be scientifically unjustified. The updated reference API
instead provides a reproducible test route using a solver already present in
the project, with preserved original RHS functions and explicit limitations.

The source `network_params.py` uses a separate `CE.timescale=0.001` while
standard neural class constructors use `self.timescale=0.01`. The **actual
instantiated model class** determines the original PPC time indexing. The new
reference wrapper prohibits PPC coupling-step refinement until that time
semantics is addressed deliberately in the later delay milestone.

The repository's neuronal units PDF provides neural units, but the existing
body/contact values are not accompanied by an independently verified common
SI-unit specification. We have therefore **not** labeled the contact penalty
force calibration as biologically validated.
