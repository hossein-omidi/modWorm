"""
modWorm: Modular simulation of neural connectomics, dynamics and biomechanics of Caenorhabditis elegans
Copyright (c) 2024-2025 University of Washington. Developed in UW NeuroAI Lab by Jimin Kim.
"""

__author__ = 'Jimin Kim: jk55@u.washington.edu'
__version__ = '1.0.42'

import importlib.util
import json
import os
import sys


def _conda_julia_project():
    if os.environ.get("JULIA_PROJECT") or os.environ.get("PYTHON_JULIACALL_PROJECT"):
        return None

    prefix = sys.prefix
    if sys.prefix == sys.base_prefix:
        prefix = os.environ.get("CONDA_PREFIX", prefix)
    project = os.path.join(prefix, "julia_env", "Project.toml")
    return os.path.dirname(project) if os.path.isfile(project) else None

class JuliaBackendUnavailableError(ImportError):
    """Raised when a Julia-backed operation is requested without juliacall."""


try:
    JULIA_AVAILABLE = importlib.util.find_spec("juliacall") is not None
except (ImportError, ValueError):
    JULIA_AVAILABLE = False

_JuliaMain = None
_julia_import_error = None


class _JuliaMainAdapter:
    """Defer Julia initialization until a caller explicitly selects that backend."""

    def __init__(self):
        self._loading_python_modules = True
        self._deferred_sources = []
        self._initialized_sources = 0

    def eval(self, source):
        if self._loading_python_modules:
            self._deferred_sources.append(source)
            return None
        self.ensure_ready()
        return _JuliaMain.seval(source)

    def _finish_python_import(self):
        self._loading_python_modules = False

    def ensure_ready(self):
        global JULIA_AVAILABLE, _JuliaMain, _julia_import_error
        if not JULIA_AVAILABLE:
            raise JuliaBackendUnavailableError(
                "Julia-specific functionality requires juliacall and the modWorm "
                "Julia packages; the Python simulation backend remains available."
            ) from _julia_import_error

        try:
            if _JuliaMain is None:
                from juliacall import Main as julia_main

                _JuliaMain = julia_main
            if self._initialized_sources == 0:
                project = _conda_julia_project()
                if project is not None:
                    project_literal = json.dumps(project)
                    _JuliaMain.seval(
                        f"import Pkg; Pkg.activate({project_literal})"
                    )
            while self._initialized_sources < len(self._deferred_sources):
                source = self._deferred_sources[self._initialized_sources]
                _JuliaMain.seval(source)
                self._initialized_sources += 1
        except Exception as exc:
            _julia_import_error = exc
            if _JuliaMain is None:
                JULIA_AVAILABLE = False
            raise JuliaBackendUnavailableError(
                "The Julia runtime or modWorm Julia dependencies could not be "
                "initialized. Install juliacall and the Julia packages listed in "
                "README.md before selecting a Julia-backed operation."
            ) from exc

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        self.ensure_ready()
        return getattr(_JuliaMain, name)


Main = _JuliaMainAdapter()


def require_julia():
    """Raise an actionable error if a caller selects a Julia-only operation."""
    Main.ensure_ready()

if JULIA_AVAILABLE:
    Main.eval("""

        using DifferentialEquations, OrdinaryDiffEq, Sundials, SciMLBase, LinearAlgebra, LogExpFunctions, Interpolations, StatsBase, Statistics

        function modworm_validate_solution(sol, tf, nsteps, label)
            SciMLBase.successful_retcode(sol) || error("$label solver failed: $(sol.retcode)")
            length(sol.t) == nsteps || error("$label saved $(length(sol.t)) of $nsteps requested samples")
            isapprox(sol.t[end], tf; rtol=1e-10, atol=1e-12) ||
                error("$label stopped at t=$(sol.t[end]); expected t=$tf")
            all(state -> all(isfinite, state), sol.u) ||
                error("$label returned a non-finite state")
            return sol
        end

        """)

from modWorm import sys_paths                  # File paths needed for simulations

from modWorm import network_params             # Predefined neural parameters for animals
from modWorm import network_dynamics           # Predefined set of equations (Python + Julia) for individual neuron dynamics (ion channels, leak channel etc)
from modWorm import network_interactions       # Predefined set of equations (Python + Julia) for neuronal interactions through connectome (synaptic, gap, etc)
from modWorm import network_simulations        # Master functions (Python + Julia) for simulating nervous system

from modWorm import muscle_body_params         # Predefined muscles/body parameters for animals
from modWorm import muscle_dynamics            # Predefined set of equations (Python + Julia) for neurons -> muscle translations
from modWorm import body_dynamics              # Predefined set of equations (Python + Julia) for viscoelastic model
from modWorm import body_simulations           # Master functions (Python + Julia) for simulating body from neural activities

from modWorm import proprioception_simulation  # Master functions (Python + Julia) for simulating nervous system + body with proprioception, supports multi-threaded simulations

from modWorm import animation                  # Functions for animating body simulations
from modWorm import utils                      # Utility functions

from modWorm import predefined_classes_nv      # Predefined set of nervous system classes 
from modWorm import predefined_classes_mb      # Predefined set of muscle/body classes

Main._finish_python_import()

from modWorm import numerics
from modWorm import collective
from modWorm import shared_world