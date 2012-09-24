"""
This test is intended to be a verification of the splitting solver for
the bidomain equations plus FitzHugh-Nagumo model to be compared with
some known code, for instance PyCC

Data and parameters suggested by Glenn T. Lines, 22. sept 2012,
also match Table 2.1 p 29 in Sundnes et al.
"""

# Marie E. Rognes <meg@simula.no>
# Last changed: 2012-09-24

import math

from dolfin import *
from beatadjoint import *
from beatadjoint.models import *

class InitialCondition(Expression):
    def eval(self, values, x):
        r = math.sqrt(x[0]**2 + x[1]**2)
        values[1] = 0.0
        if r < 0.25:
            values[0] = 30.0
        else:
            values[0] = -85.0
    def value_shape(self):
        return (2,)

class MyHeart(CardiacModel):
    def __init__(self, cell_model):
        CardiacModel.__init__(self, cell_model)
    def domain(self):
        return UnitSquare(100, 100)
    def conductivities(self):
        chi = 2000.0   # cm^{-1}
        s_il = 3.0/chi # mS
        s_it = 0.3/chi # mS
        s_el = 2.0/chi # mS
        s_et = 1.3/chi # mS
        M_i = as_tensor(((s_il, 0), (0, s_it)))
        M_e = as_tensor(((s_el, 0), (0, s_et)))
        return (M_i, M_e)

# Set-up cell model
k = 0.00004;
Vrest = -85.;
Vthreshold = -70.;
Vpeak = 40.;
k = 0.00004;
l = 0.63;
b = 0.013;
v_amp = Vpeak - Vrest
cell_parameters = {"c_1": -k*v_amp**2, "c_2": k*v_amp, "c_3": b/l,
                   "a": (Vthreshold -Vrest)/v_amp, "b": l,
                   "v_rest":Vrest, "v_peak": Vpeak}
cell = FitzHughNagumo(cell_parameters)

# Set-up cardiac model
heart = MyHeart(cell)

# Set-up solver
application_parameters = Parameters()
application_parameters.add("theta", 0.5)
application_parameters.add("enable_adjoint", False)
application_parameters.add("store_solutions", True)
application_parameters.add("plot_solutions", True)
solver = SplittingSolver(heart, application_parameters)

# Define end-time and (constant) timestep
dt = 0.25 # mS
T = 100   # mS

# Define initial condition(s)
ic = InitialCondition()
vs0 = project(ic, solver.VS)
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
