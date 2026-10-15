"""Milestone 2 reference-integration script for modWorm.
Reuses original simulation entry points (proprioception_simulation.run_network,
shared_world.run_single_worm / run_two_worm). No new biological equations.
Reference contract: algebraic muscle mapping preserved; b=[q,qdot] preserved;
neural RHS preserved; discrete PDF update preserved; discrete proprioceptive delay preserved.
"""
import time, sys, traceback
import numpy as np

# Ensure modWorm imports work
sys.path.insert(0, '.')

from modWorm import predefined_classes_nv as nv_classes
from modWorm import predefined_classes_mb as mb_classes
from modWorm import proprioception_simulation as prop_sim
from modWorm import shared_world
from modWorm import network_params as n_params
from modWorm import muscle_body_params as mb_params
from modWorm import muscle_dynamics as m_dyn
from modWorm import body_dynamics as b_dyn
from modWorm import collective

# Load standard connectome data (existing files)
import numpy as np
gap_conn = np.load('modWorm/data/conn_gap_adjust_Cook.npy')  # or default from presets
syn_conn = np.load('modWorm/data/conn_syn_adjust_Cook.npy')

# Note: The preset inputs reference the standard network with these connections.
# For reproducibility we use the default gap/syn matrices loaded from files.
# If the standard tutorial uses predefined matrices, we fall back to the params.
