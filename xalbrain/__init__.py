"""
The xalbrain Python module is a problem and solver collection for
cardiac electrophysiology models.

To import the module, type::

  from xalbrain import *

"""

# Model imports
from xalbrain.cellmodels import *

from xalbrain.markerwisefield import rhs_with_markerwise_field
from xalbrain.cardiacmodels import CardiacModel

# Solver imports
from xalbrain.splittingsolver import (
    BasicSplittingSolver,
    SplittingSolver
)
from xalbrain.cellsolver import (
    BasicSingleCellSolver,
    SingleCellSolver,
    BasicCardiacODESolver,
    CardiacODESolver
)

from xalbrain.bidomainsolver import (
    BasicBidomainSolver,
    BidomainSolver
)

from xalbrain.monodomainsolver import (
    BasicMonodomainSolver,
    MonodomainSolver
)

# Various utility functions, mainly for internal use
from xalbrain.utils import (
    splat,
    state_space,
    end_of_time,
    Projecter
)

from xalbrain.better_odesolver import BetterODESolver
