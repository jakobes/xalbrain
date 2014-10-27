"""
The beatadjoint Python module is a problem and solver collection for
cardiac electrophysiology models.

To import the module, type::

  from beatadjoint import *

"""

import dolfinimport

# Import all of dolfin with possibly dolfin-adjoint on top
from dolfinimport import *

# Model imports
from beatadjoint.cardiacmodels import CardiacModel
from beatadjoint.cellmodels import *

# Solver imports
from beatadjoint.splittingsolver import BasicSplittingSolver
from beatadjoint.splittingsolver import SplittingSolver
from beatadjoint.cellsolver import BasicSingleCellSolver
from beatadjoint.cellsolver import BasicCardiacODESolver, CardiacODESolver
from beatadjoint.bidomainsolver import BasicBidomainSolver
from beatadjoint.bidomainsolver import BidomainSolver
from beatadjoint.monodomainsolver import BasicMonodomainSolver
from beatadjoint.monodomainsolver import MonodomainSolver

# Various utility functions, mainly for internal use
import beatadjoint.utils

from beatadjoint.timeseries import HDF5TimeSeries

# Set-up some global parameters
beat_parameters = dolfinimport.Parameters("beat-parameters")
beat_parameters.add("enable_adjoint", True)
