""" The xalbrain Python module is a problem and solver collection for cardiac electrophysiology models."""

from xalbrain.brainmodel import BrainModel

from xalbrain.odesolver import (
    SubDomainODESolver,
    ODESolver,
    SubDomainODESolver,
    SingleCellSolver,
    ODESolverParameters,
)

from xalbrain.bidomain import (
    BidomainSolver,
    BidomainParameters,
)

from xalbrain.monodomain import (
    MonodomainSolver,
    MonodomainParameters,
)

# Solver imports
from xalbrain.splittingsolver import (
    MonodomainSplittingSolver,
    BidomainSplittingSolver,
    BidomainSplittingSolverSubDomain,
    SplittingSolverParameters,
)

# Various utility functions, mainly for internal use
import xalbrain.utils

# NB: Workaround for FEniCS 1.7.0dev
import ufl
ufl.algorithms.apply_derivatives.CONDITIONAL_WORKAROUND = True
