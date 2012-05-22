from dolfin import *
from dolfin_adjoint import *
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
solver = SplittingSolver(heart, parameters)

# Set initial conditions
v_expr = Expression("- x[0]*(1-x[0])*x[1]*(1-x[1])")
v_ = project(v_expr, solver.V)
(v, u, s) = solver.solution_fields()
v.assign(v_)

solver.solve((0, 0.1), 0.01)

# Just quick regression test, not validation
ref =  0.028030779172955524
a = norm(v)
assert abs(a - ref) < 1.e-9, "a = %g" % a
