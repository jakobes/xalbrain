"""
This test just solves the bidomain equations with an analytic solution
(assuming no state variables) to verify the correctness of the
splitting solver.
"""
# Marie E. Rognes <meg@simula.no>
# Last changed: 2012-09-19

from dolfin import *

# Cardiac solver specific imports
from beatadjoint import *
from beatadjoint.models import *

level = 1
set_log_level(ERROR)

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
heart.applied_current = Expression(ac_str, t=0, degree=5)

application_parameters = Parameters()
application_parameters.add("theta", 0.5)
application_parameters.add("enable_adjoint", False)

solver = SplittingSolver(heart, application_parameters)

T = 0.1
dt = 0.01/(2**level)

v_exact = Expression("cos(2*pi*x[0])*cos(2*pi*x[1])*sin(t)", t=T, degree=5)
u_exact = Expression("-cos(2*pi*x[0])*cos(2*pi*x[1])*sin(t)/2.0", t=T, degree=5)

plot(v_exact, title="v_exact", mesh=heart.domain())
plot(u_exact, title="u_exact", mesh=heart.domain())

# Define initial condition(s)
vs0 = Function(solver.VS)

(vs_, vs, u) = solver.solution_fields()
vs_.assign(vs0)

# Run main stuff
info_green("Solving primal")
solver.solve((0, T), 0.01)
(v, s) = vs.split()

plot(v, title="v")
plot(u, title="u")

v_error = errornorm(v_exact, v, "L2", degree_rise=5)
u_error = errornorm(u_exact, u, "L2", degree_rise=5)
print "v_error = ", v_error
print "u_error = ", u_error
interactive()
