# Serotonin modulation

This optional layer adds a MOD-1 chloride current to the existing neural voltage
equation. It does not change the synaptic or gap-junction connectomes, muscle
model, biomechanics, or proprioceptive feedback.

The source is externally controlled; it is not an additional ODE state. Supply
either a fixed concentration or a callable that accepts simulation time, in
micromolar:

```python
from modWorm.serotonin import SerotoninModulation

nn.serotonin_modulation = SerotoninModulation(concentration=5.0)
# Or, for an externally specified time-varying control:
nn.serotonin_modulation.set_concentration(lambda t: 5.0 if t >= 2.0 else 0.0)
```

`SerotoninModulation` computes the Hill steady state and MOD-1 current. The
`receptor_activation` method is an override point for a kinetic receptor
implementation. The supplied Python network classes pass simulation time to
the current calculation, subtract `I_MOD-1` in the voltage equation, and
include the current's voltage derivative in their analytic Jacobian.

Defaults: `g_mod1=2 nS`, `k_d=1 uM`, `hill_coefficient=1`, and `e_cl=-70 mV`.
Only AIYL (index 137) and AIYR (index 133) are targets by default. Optional
targets `ASIL`/`ASIR` (75/89) and `RIFL`/`RIFR` (158/148) can be supplied as
`target_neurons`. Indices are zero-based and come from `modWorm/neurons.json`.
That 279-neuron list does not contain an NSM entry.

With a missing modulation object, a disabled layer, or zero serotonin, the
voltage equation evaluates its original RHS expression unchanged. The test script
in `examples/test_serotonin.py` compares zero and 5 uM full-model simulations
using the Python solvers. The Julia-compiled network classes do not currently
consume this Python control object; use the standard Python
`CelegansWorm_NervousSystem` class with `network_simulations.run_network` and
`body_simulations.run_body` for the end-to-end serotonin comparison.
