# Neuromodulation and collective-behavior research extension

## Current implementation boundary

The extension is deliberately staged. It adds analysis and mechanism primitives;
it does **not** yet claim a validated roaming/dwelling model or a coupled
multi-worm simulation.

| Component | Implemented | Scientific / engineering status |
|---|---|---|
| MOD-1 | External concentration, Hill occupancy, chloride-current term, selectable targets | MOD-1 is a ligand-gated chloride channel. Default 2 nS, 1 µM and -70 mV values are exploratory, not calibrated. |
| Neuron identities | Loaded from `modWorm/neurons.json` | The current 279-neuron list has no NSM node; no NSM node or edges are synthesized. |
| PDFR-1 | External PDF-1, Hill occupancy, first-order abstract cAMP-like state | GPCR/cAMP representation is a reduced hypothesis. Default affinity, time constant and gain are assumptions. |
| PDF downstream effect | Optional signed bias current in the Python neural RHS, zero gain by default | Explicitly phenomenological and not an established PDFR-1 current law. |
| Locomotion analysis | Signed axial velocity, bout segmentation, reversals, turns, curvature and path metrics | Observational only; it never controls the simulation. A caller may reverse the assumed head/tail ordering. |
| Group analysis | Distance-graph clusters, pairwise distances, encounter durations and global density | Geometric proximity is not evidence for attraction or social intent. |
| Body contact | Capsule-penalty contact forces on segmented centerlines | Equal-and-opposite nodal forces are computed, but they are not yet mapped into the body's generalized-coordinate ODE. |
| Shared world | Not implemented | Independent, synchronized, mechanically coupled worms remain a subsequent stage. |

## Scientific interpretation

| Mechanism / claim | Evidence status | Implementation consequence |
|---|---|---|
| MOD-1 is a serotonin-gated chloride channel; serotonin contributes to food-foraging state regulation | Experimentally established in the cited Flavell et al. (2013) result and source brief | Model the receptor as a chloride conductance, not an excitatory current source. |
| AIY, ASI and RIF participate in serotonin-dependent regulation | Functionally implicated in Flavell et al. (2013); expression/participation do not imply invented synapses | These are configurable MOD-1 targets; AIY remains the default. |
| PDF-1/PDFR-1 is associated with roaming and persistent-state circuitry; PDFR-1 is GPCR/cAMP-related | Functionally supported in Flavell et al. (2013) and Ji et al. (2021); the prompt supplies the Gs/cAMP rationale | Track receptor occupancy and a slow abstract signal separately from MOD-1. |
| Cell-specific PDFR-1 electrophysiological effect | Not uniquely identified by the supplied evidence | Optional signed current is marked as a hypothesis, is zero by default, and must be calibrated before interpreting behavior. |
| NSM source role | Biologically supported, but NSM is absent from this model's 279 entries | No NSM neuron or anatomical edge is invented. |
| CASY-1, neuromodulation and swarming on food lawns | Context-specific evidence from Shahi et al. (2026) in the prompt; intermediate causal chain remains partly unresolved | Do not interpret reduced-order liquid swimming or proximity clusters as a reproduction of agar swarming. |

The cited source brief supports serotonin/MOD-1 involvement in food-foraging
roaming and dwelling and PDF-1/PDFR-1 involvement in persistent
roaming-associated signaling. It does not identify a universal
PDFR-1-to-current equation, and the reported agar/food-lawn results must not be
treated as validated swimming predictions. The PDF module reports receptor
occupancy and a slow normalized signaling proxy. An explicitly configured
signed bias can affect selected PDFR-1 target neurons in the Python
proprioceptive solver, but the evidence does not establish its sign or
magnitude.

The MOD-1 current is expressed as `g * occupancy * (V - E_Cl)` and subtracted
from the modWorm voltage RHS. The effect is not guaranteed to be hyperpolarizing
when membrane voltage is below `E_Cl`; chloride-channel activation can also
shunt. Parameter sensitivity is required because the default conductance
greatly exceeds the original leak conductance.

Current exploratory defaults are MOD-1 `g=2 nS`, `EC50=1 µM`, Hill coefficient
`n=1` and `E_Cl=-70 mV`; the 5 µM dose is an intervention, not a calibrated
physiological concentration. PDFR-1 defaults are `K_P=1 µM`, `m=1`,
`tau_c=5 s`, `alpha=1` normalized unit and downstream bias gain `0 pA/unit`.
Only the MOD-1 EC50-like value is presented as literature-motivated in the
original serotonin README; the conductance and all PDFR-1 dynamics/coupling
parameters remain assumed exploratory settings.

The current Python proprioceptive path uses delayed projected neural activity
(`fwd_i_ProprioceptiveDelayed`), not direct body stretch. Existing Julia network
classes do not consume the Python modulation objects. Use Python classes for
MOD-1 tests and do not mix backend results as if equivalent.

## API examples

```python
import numpy as np

from modWorm.serotonin import SerotoninModulation
from modWorm.pdf_modulation import PDFR1Modulation

# Existing Python neural-network classes consume this MOD-1 term.
nn.serotonin_modulation = SerotoninModulation(
    concentration=lambda t: 2.5 * (1.0 - np.cos(2.0 * np.pi * t / 20.0)),
    target_neurons=("AIYL", "AIYR"),
)

# PDFR-1/cAMP is independent. Nonzero coupling is an explicit hypothesis.
nn.pdfr1_modulation = PDFR1Modulation(
    concentration=5.0,
    coupling_gain_pA_per_signal=0.0,
)
```

`PDFR1Modulation.advance` assumes occupancy is constant at the time-interval
midpoint and exactly advances its first-order state for that occupancy. The
state is normalized and has arbitrary units. `proprioception_simulation.run_network`
advances this state with its network timestep and includes the configured
phenomenological current in Python network classes. Feed-forward integration
through `network_simulations.run_network` does not advance this state; callers
must provide a synchronized update strategy before using that route for
time-varying PDF-1 experiments. The Julia classes do not consume either module.
Setting the coupling gain to zero is the default and leaves neuronal equations
unchanged.

`modWorm.behavior_analysis.locomotion_metrics` accepts sampled time and body
centerlines. It reports signed velocity projected onto head-to-tail orientation,
forward/backward/neutral bouts, direction changes, angular turns, discrete
curvature, path length, net displacement and convex-hull area. Thresholds are
classification settings, not biological calibration. No roaming/dwelling label
is emitted by default.

`modWorm.behavior_analysis.population_metrics` accepts centroid trajectories
with shape `(time, worms, 2)` and an explicit distance threshold. Clusters are
connected components of a distance graph; `domain_area`, if supplied, defines
the reported global worms-per-area density.

`modWorm.collective.capsule_contact_forces` returns nodal contact forces for two
sampled centerlines. It applies a unilateral linear penalty and closing-only
damping and enforces action/reaction. Contact stiffness and damping must be
chosen relative to body scale, timestep and integration stability. Applying
these forces requires a separate, validated force-to-generalized-coordinate
adapter; plotting these forces alone is not a contact-coupled simulation.

## Validation and reproducibility

Run focused tests with the project Python environment:

```powershell
$env:PATH = 'C:\Users\ecer\.conda\envs\celegan;C:\Users\ecer\.conda\envs\celegan\Scripts;' + $env:PATH
python -m unittest discover -s tests -p 'test_*.py' -v
```

The tests cover MOD-1 targeting/sign/zero-dose behavior and Jacobian entries,
canonical neuron indices, PDFR-1 dose response and state isolation, observational
reversal and cluster metrics, action/reaction contact, and the supported stiff
body-stepper configuration. The synaptic logistic and its analytic derivative
use a numerically stable sigmoid to avoid overflow at extreme voltages without
changing the mathematical response. Tests do not establish full-duration
locomotion effects, biological parameter calibration, numerical convergence of
a shared world, or collective swarming.

The matched baseline/zero/5 µM integration diagnostic can be run from the
repository root:

```powershell
python examples/test_serotonin.py database\NeuronConnect.xls database\NeuronFixedPoints.xls modWorm\presets_input\input_mat_gentle_post_touch.npy --max-steps 12
```

Omit `--max-steps 12` to use the complete input. `--seed` defaults to
`20261014`. The script reports AIY sigmoid drive and body-center speed without
asserting that either must decrease.

The full-body `proprioception_simulation.run_network` route was measured to be
too expensive for a multi-condition interactive sweep on the tested local
machine (the 13.99 s notebook run exceeded a one-hour execution limit). Begin
with short simulations and establish a measured runtime before launching
population experiments. A 12-sample transient smoke test gave different AIY
mean voltages and signed centerline velocities at baseline and 5 µM, but its
0.11 s window is too short and numerically unrepresentative to infer a
roaming/dwelling response or a locomotor trend.
