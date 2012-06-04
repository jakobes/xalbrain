from dolfin import *
from dolfin_adjoint import *

# Cardiac solver specific imports
from models import *
from cellsolver import *

cell = FitzHughNagumo()

solver = CellSolver(cell)

(vs_, vs) = solver.solution_fields()

solver.solve((0, 0.1), 0.01)

(v, s) = vs.split()

plot(v, interactive=True)
plot(s, interactive=True)
