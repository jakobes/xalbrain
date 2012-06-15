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

level = 2

class MyHeart(CardiacModel):
    def __init__(self, cell_model):
        CardiacModel.__init__(self, cell_model)

    def domain(self):
        N = 10*(2**level)
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

T = 0.1
dt = 0.01/(2**level)

v_exact = Expression("cos(2*pi*x[0])*cos(2*pi*x[1])*sin(t)", t=T, degree=3)
u_exact = Expression("-cos(2*pi*x[0])*cos(2*pi*x[1])*sin(t)/2.0", t=T, degree=3)

def main(vs0):
    (vs_, vs, u) = solver.solution_fields()
    vs_.assign(vs0, annotate=application_parameters["enable_adjoint"])
    solver.solve((0, T), 0.01)
    return (vs, vs_, u)

# Define initial conditions
vs0 = Function(solver.VS)

# Run main stuff
info_green("Solving primal")
(vs, vs_, u) = main(vs0)

(v, s) = vs.split()

v_error = errornorm(v_exact, v, "L2")
u_error = errornorm(u_exact, u, "L2")
print "v_error = ", v_error
print "u_error = ", u_error

#interactive()
