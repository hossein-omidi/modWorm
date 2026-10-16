"""A sub-second Julia-side type check AFTER initial Julia compilation.

Run from the modWorm repository root:
    python examples/verify_julia_native_bridge.py
No ODE integration or biological simulation is performed.
"""
import numpy as np
from modWorm import Main, require_julia

require_julia()
Main.eval('''
function modworm_native_bridge_test(params)
    d = modworm_native_ppc_dict(params)
    return (d["initial"] isa Vector{Float64}) &&
           (d["matrix"] isa Matrix{Float64}) &&
           (size(d["matrix"]) == (2,3)) &&
           (d["matrix"][2,3] == 5.0) &&
           (d["initial"][1] == 1.0)
end
''')

params = {
    'initial': np.array([1.0, 2.0, 3.0], dtype=np.float64),
    'matrix': np.arange(6.0, dtype=np.float64).reshape(2, 3),
}
verified = bool(Main.modworm_native_bridge_test(params))
if not verified:
    raise RuntimeError('Julia native array bridge failed its type or value-preservation check')
print('PASS: NumPy vectors and matrices become Julia-owned arrays with unchanged values')
