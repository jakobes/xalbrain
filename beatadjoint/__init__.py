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
from beatadjoint.splittingsolver import SplittingSolver
from beatadjoint.fullycoupledsolver import CoupledBidomainSolver
from beatadjoint.cellsolver import BasicSingleCellSolver
from beatadjoint.cellsolver import BasicCardiacODESolver

# Various utility functions, mainly for internal use
import beatadjoint.utils
