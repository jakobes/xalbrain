"""
This test is intended to be a verification of the splitting solver for
the bidomain equations plus FitzHugh-Nagumo model to be compared with
some known code, for instance PyCC
"""
# Marie E. Rognes <meg@simula.no>
# Last changed: 2012-09-19

from dolfin import *
from beatadjoint import *
from beatadjoint.models import *

class AppliedCurrent(Expression):
    def __init__(self, t=0.0):
        self.t = t
    def eval(self, value, x):
        if self.t >= 10 and self.t < 20:
            v_amp = 125
            value[0] = 0.05*v_amp*10*exp(-(pow(x[0], 2) + pow(x[1] - 0.5, 2)) / 0.02)
        else:
            value[0] = 0.0

class MyHeart(CardiacModel):
    def __init__(self, cell_model):
        CardiacModel.__init__(self, cell_model)
    def domain(self):
        N = 30
        return UnitSquare(N, N)
    def conductivities(self):
        M_i = 1.0
        M_e = 2.0
        return (M_i, M_e)

# Set-up model
cell = FitzHughNagumo()
cell.applied_current = AppliedCurrent()
heart = MyHeart(cell)

# Set-up solver
application_parameters = Parameters()
application_parameters.add("theta", 1.0)
application_parameters.add("enable_adjoint", False)
application_parameters.add("store_solutions", True)
solver = SplittingSolver(heart, application_parameters)

# Define end-time and (constant) timestep
dt = 1.
T = 100

# Define initial condition(s)
vs0 = Expression(("-85.0", "0.0"))
vs0 = project(vs0, solver.VS)
(vs_, vs, u) = solver.solution_fields()
vs_.assign(vs0)

# Solve
info_green("Solving primal")
solver.solve((0, T), dt)
(v, s) = vs.split()

plot(v, title="v")
plot(s, title="s")
plot(u, title="u")
interactive()
