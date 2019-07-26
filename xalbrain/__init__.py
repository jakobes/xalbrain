""" The xalbrain Python module is a problem and solver collection for cardiac electrophysiology models."""

# Model imports
from xalbrain.cardiacmodels import CellModel
from xalbrain.cellmodels import *

# Solver imports
from xalbrain.splittingsolver import (
    MonodomainSplittingSolver,
    BidomainSplittingSolver
)

from xalbrain.odesolver import (
    SubDomainODESolver,
    ODESolver,
    SubDomainODESolver,
    SingleCellSolver,
)

from xalbrain.bidomain import BidomainSolver
from xalbrain.monodomain import MonodomainSolver

# Various utility functions, mainly for internal use
import xalbrain.utils

# NB: Workaround for FEniCS 1.7.0dev
import ufl
ufl.algorithms.apply_derivatives.CONDITIONAL_WORKAROUND = True
