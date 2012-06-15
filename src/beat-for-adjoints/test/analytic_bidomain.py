"""
This test just solves the bidomain equations with an analytic solution
(assuming no state variables) to verify the correctness of the
splitting solver.
"""

from dolfin import *
from dolfin_adjoint import *

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

cell = NoCellModel()
heart = MyHeart(cell)
ac_str = "cos(t)*cos(2*pi*x[0])*cos(2*pi*x[1]) + 4*pow(pi, 2)*cos(2*pi*x[0])*cos(2*pi*x[1])*sin(t)"
heart.applied_current = Expression(ac_str, t=0)

application_parameters = Parameters()
application_parameters.add("theta", 0.5)
application_parameters.add("enable_adjoint", False)

solver = SplittingSolver(heart, application_parameters)

def main(vs0):
    (vs_, vs, u) = solver.solution_fields()
    vs_.assign(vs0, annotate=application_parameters["enable_adjoint"])
    solver.solve((0, 1.0), 0.01)
    return (vs, vs_)

# Define initial conditions
vs0 = Function(solver.VS)

# Run main stuff
info_green("Solving primal")
(vs, vs_) = main(vs0)

interactive()
