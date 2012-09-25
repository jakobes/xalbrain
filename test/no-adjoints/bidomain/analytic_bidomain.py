"""
This test just solves the bidomain equations with an analytic solution
(assuming no state variables) to verify the correctness of the
splitting solver.
"""
# Marie E. Rognes <meg@simula.no>
# Last changed: 2012-09-25

from dolfin import *
from beatadjoint import *

level = 0
#set_log_level(ERROR)

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

# Set-up model
cell = NoCellModel()
heart = MyHeart(cell)
ac_str = "cos(t)*cos(2*pi*x[0])*cos(2*pi*x[1]) + 4*pow(pi, 2)*cos(2*pi*x[0])*cos(2*pi*x[1])*sin(t)"
heart.applied_current = Expression(ac_str, t=0, degree=5)

# Set-up solver
parameters = SplittingSolver.default_parameters()
parameters["linear_variational_solver"]["linear_solver"] = "direct"
solver = SplittingSolver(heart, parameters)
theta = solver.parameters["theta"]

# Define end-time and (constant) timestep
T = 0.1
dt = 0.01/(2**level)

# Define exact solution (Note: v is returned at end of time
# interval(s), u is computed at somewhere in the time interval
# depending on theta)
v_exact = Expression("cos(2*pi*x[0])*cos(2*pi*x[1])*sin(t)", t=T, degree=5)
u_exact = Expression("-cos(2*pi*x[0])*cos(2*pi*x[1])*sin(t)/2.0",
                     t=T - (1 - theta)*dt, degree=5)

# Define initial condition(s)
vs0 = Function(solver.VS)
(vs_, vs, u) = solver.solution_fields()
vs_.assign(vs0)

# Solve
info_green("Solving primal")
solutions = solver.solve((0, T), dt)
for (timestep, vs, u) in solutions:
    continue

(v, s) = vs.split()

# Procomputed reference errors (for regression checking):
v_reference = 4.3105092332652306e-03
u_reference = 2.0258311577533851e-03

# Compute errors
v_error = errornorm(v_exact, v, "L2", degree_rise=5)
u_error = errornorm(u_exact, u, "L2", degree_rise=5)
v_diff = abs(v_error - v_reference)
u_diff = abs(u_error - u_reference)
tolerance = 1.e-10
msg = "Maximal %s value does not match reference: diff is %.16e"
assert (v_diff < tolerance), msg % ("v", v_diff)
assert (u_diff < tolerance), msg % ("u", u_diff)
