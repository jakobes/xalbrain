from dolfin import *
from splittingsolver import *
from models import *

domain = UnitSquare(16, 16)

cell_parameters = {"epsilon": 0.01, "gamma": 0.5, "alpha": 0.1}
cell = FitzHughNagumo(cell_parameters)

heart = CardiacModel(domain, cell)

solver = SplittingSolver(heart)

(v, u, s) = solver.solution_fields()

solver.step((0, 0.1), (v, s))
