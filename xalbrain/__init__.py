"""
The xalbrain Python module is a problem and solver collection for
cardiac electrophysiology models.

To import the module, type::

  from xalbrain import *

"""

# Import all of dolfin with possibly dolfin-adjoint on top
from xalbrain.dolfinimport import *

# Model imports
from xalbrain.cardiacmodels import CardiacModel
from xalbrain.cellmodels import *
from xalbrain.markerwisefield import *

# Solver imports
from xalbrain.splittingsolver import SplittingSolver

from xalbrain.cellsolver import (
    BasicCardiacODESolver,
    CardiacODESolver,
    BasicSingleCellSolver,
    SingleCellSolver
)

from xalbrain.bidomainsolver import (
    BasicBidomainSolver,
    BidomainSolver
)

from xalbrain.monodomainsolver import (
    BasicMonodomainSolver,
    MonodomainSolver
)

from xalbrain.parameters import (
    BidomainParameters,
    MonodomainParameters,
    SingleCellParameters,
    SplittingParameters,
    KrylovParameters,
    LUParameters,
)

# Various utility functions, mainly for internal use
import xalbrain.utils

# # NB: Workaround for FEniCS 1.7.0dev
# import ufl
# ufl.algorithms.apply_derivatives.CONDITIONAL_WORKAROUND = True

# Set-up some global parameters
beat_parameters = dolfinimport.Parameters("beat-parameters")
beat_parameters.add("enable_adjoint", False)
