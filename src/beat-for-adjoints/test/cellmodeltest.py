from dolfin import *
from dolfin_adjoint import *

# Cardiac solver specific imports
from models import *
from cellsolver import *

cell = FitzHughNagumo()

solver = CellSolver(cell)

# Setup initial condition
(vs_, vs) = solver.solution_fields()
vs_.vector()[0] = -85. # Initial condition resting state
vs_.vector()[1] = 0.

solver.solve((0, 0.1), 0.01)

(v, s) = vs.split()

