from dolfin import *

# Cardiac solver specific imports
from splittingsolver import *
from models import *

class MyHeart(CardiacModel):
    def __init__(self, cell_model):
        CardiacModel.__init__(self, cell_model)

    def domain(self):
        N = 16
        return UnitSquare(N, N)

    def conductivities(self):
        M_i = 1.0
        M_e = 1.0
        return (M_i, M_e)

cell_parameters = {"epsilon": 0.01, "gamma": 0.5, "alpha": 0.1}
cell = FitzHughNagumo(cell_parameters)
heart = MyHeart(cell)

parameters = Parameters()
parameters.add("theta", 0.5)
parameters.add("enable_adjoint", True)
solver = SplittingSolver(heart, parameters)

# Set initial conditions
vs_expr = Expression(("- x[0]*(1-x[0])*x[1]*(1-x[1])", "0.0"))
vs_ = project(vs_expr, solver.VS)
(vs, u) = solver.solution_fields()
vs.assign(vs_)

solver.solve((0, 0.1), 0.01)

# Just quick regression test, not validation
print "-"*80
ref =  0.028030779172955524
a = norm(vs.split()[0])
diff = abs(a - ref)
assert diff < 1.e-9, "a = %g, diff = %g" % (a, diff)
print "-"*80

