"""
The beatadjoint Python module is a problem and solver collection for
cardiac electrophysiology models.

To import the module, type::

  from beatadjoint import *

"""

# Model imports
from beatadjoint.cardiacmodels import CardiacModel
from beatadjoint.cellmodels import *

# Solver imports
from beatadjoint.splittingsolver import BasicSplittingSolver
from beatadjoint.cellsolver import BasicSingleCellSolver
from beatadjoint.cellsolver import BasicCardiacODESolver
from beatadjoint.bidomainsolver import BasicBidomainSolver
from beatadjoint.bidomainsolver import BidomainSolver

# Various utility functions, mainly for internal use
import beatadjoint.utils

# Set-up some global parameters
beat_parameters = Parameters("beat-parameters")
beat_parameters.add("enable_adjoint", True)
