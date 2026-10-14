# modWorm

[Official Introduction Page](https://shlizee.github.io/modWorm/)

modWorm is a generic modeling framework for modular simulation of neural connectomics, dynamics and biomechanics of Caenorhabditis elegans (C. elegans).
\
\
modWorm enables fully modular construction and simulation of C. elegans neural connectomics, dynamics, biomechanics and the environment. modWorm employs a modular hierarchical modeling pipeline where users can freely combine smaller models (e.g., individual biophysical processes) to construct larger models (e.g., nervous system) in an efficient and intuitive manner. 
\
\
The constructed model is customizable and is reusable. The underlying components can be edited or exchanged within the framework interface. modWorm has been developed as a compact all-in-one Python library with support for Julia backbone to achieve high-performance efficient simulation. The framework currently includes scenarios for simulating the nervous system, muscles and body of C. elegans as detailed in [citation below].

# Dependencies

#### Python (>3.11 recommended):
scipy, matplotlib, statsmodels, ipython, jupyter, ffmpeg, seaborn
#### Julia (>1.9 recommended):
DifferentialEquations, OrdinaryDiffEq, Sundials, LinearAlgebra, LogExpFunctions, Interpolations, StatsBase

`juliacall` is optional for the Python-only simulation paths and is needed only when selecting Julia-backed functions. Python import does not initialize Julia. A Julia-backed call explicitly checks its runtime and packages and raises `JuliaBackendUnavailableError` if they are not usable.

# Installation and Demo

You can find installation instruction for both Mac and Windows under /pdf_tutorials. Once installed, use jupyter notebook navigator to open quick start demo tutorials under /ipynb_tutorials folder. 
\
\
You can also find the tutorial slides going over the demo notebooks "modWorm quick-start.pdf" under /pdf_tutorials.

# Python numerical and shared-world simulations

Python integrations validate solver success, requested sample times, state dimensions, and finite values. Incomplete integrations raise `NumericalIntegrationError` instead of returning uncomputed zero-filled samples.

`modWorm.shared_world` provides `WormAgent`, `run_single_worm`, `run_two_worms`, and `run_shared_world` for any positive agent count. Each agent retains independent neural, muscle, body, and optional PDFR-1 state. Pairwise capsule contacts are culled by expanded body bounds, then accumulated inside the synchronized body ODE; mechanical contact and contact-derived touch-cell currents can be enabled independently. The analytical neural Jacobian uses the same contact current as the right-hand side when that current changes the synaptic threshold. The contact stiffness, touch-cell locations, and current gain remain exploratory parameters, not calibrated biological measurements; neural and body updates are operator-split on the shared sample clock.

The automated shared-world tests execute four agents over one 0.01 s interval with contact disabled (using lightweight test neural dynamics and the original body model) and a three-agent, two-contact force-accumulation case. The four-agent test took 41.7 s in the selected Windows environment, so this is functional coverage rather than a scalability result. Full-connectome multi-agent scalability above the previously measured two-worm quick run, including eight worms, has not been benchmarked. The current contact node-to-body-force adapter has action/reaction, moment-balance, and net center-of-mass force tests, but has not yet been validated for generalized torque consistency or calibrated physical units.

Run `neuromodulation_collective_locomotion_experiment.ipynb` for quick or longer full-connectome comparisons of baseline locomotion, serotonin controls, the PDF-1 signaling hypothesis, independent worms, mechanical contact, mechanosensory contact, and observational two-worm proximity/clustering metrics. The notebook produces its trajectories and neural, muscle, body, and force measurements by executing these Python modules. It does not claim persistent roaming/dwelling states or population-level attraction.

On Windows in the `celegan` environment (Python 3.14.8, SciPy 1.18.1), the final quick full-connectome notebook baseline with six samples over 0.05 s completed in 6.10 s and used 880 body RHS evaluations across five intervals. Profiling a 0.01 s full-connectome neural/muscle/body run identified dense BDF Jacobian factorization as a major cost: converting the existing analytic Jacobian to CSC sparse form reduced its standalone neural solve from 0.804 s to 0.253 s, with identical solver counts (92 RHS evaluations, 2 Jacobians, 14 factorizations) and maximum absolute trajectory difference of `1.82e-12`. The profiled end-to-end shared-world run took 0.416 s; after the sparse change, BDF RHS evaluations and body RK45 integration remain notable costs. A representative zero-muscle 0.01 s body step used 188 RHS evaluations with RK45 versus 221 with DOP853 at `rtol=2.5e-6`, `atol=1e-3`; median times were 0.069 s and 0.073 s. Recurrence vectorization and sparse-Jacobian equivalence are tested against the original equations; these measurements do not imply biological calibration or a broad speedup.

# How to cite

If you are using this package please cite the following:
\
\
[Kim, J., Florman, J. T., Santos, J. A., Alkema, M. J., & Shlizerman, E. (2025). Modular integration of neural connectomics, dynamics and biomechanics for identification of behavioral sensorimotor pathways in Caenorhabditis elegans. bioRxiv, 724328.] [LINK TO THE PAPER](https://www.biorxiv.org/content/10.1101/724328v3)
