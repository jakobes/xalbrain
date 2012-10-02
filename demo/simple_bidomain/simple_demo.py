"""
This demo demonstrates how to set up a basic bidomain simulation
"""

# Marie E. Rognes <meg@simula.no>
# Last changed: 2012-10-02

import math
from dolfin import *
from beatadjoint import *

set_log_level(WARNING)

parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True

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

# Set-up parameters and cell model
k = 0.00004;
Vrest = -85.;
Vthreshold = -70.;
Vpeak = 40.;
k = 0.00004;
l = 0.63;
b = 0.013;
v_amp = Vpeak - Vrest
cell_parameters = {"c_1": k*v_amp**2, "c_2": k*v_amp, "c_3": b/l,
                   "a": (Vthreshold - Vrest)/v_amp, "b": l,
                   "v_rest":Vrest, "v_peak": Vpeak}
cell = FitzHughNagumo(cell_parameters)

# Set-up cardiac model
heart = MyHeart(cell)

# Set-up solver
ps = SplittingSolver.default_parameters()
ps["linear_variational_solver"]["linear_solver"] = "direct"
solver = SplittingSolver(heart, parameters=ps)

# Define end-time and (constant) timestep
dt = 0.25 # mS
T = 100.0   # mS

# Define initial condition(s)
ic = InitialCondition()
vs0 = project(ic, solver.VS)
(vs_, vs, u) = solver.solution_fields()
vs_.assign(vs0)

# Solve
info_green("Solving primal")
solutions = solver.solve((0, T), dt)

points = [(0.1, 0.1), (0.3, 0.4), (0.5, 0.5), (0.7, 0.7), (0.9, 0.9)]
v_values = []
u_values = []
for (timestep, vs, u) in solutions:
    (v, s) = vs.split()
    print "avg(u) = ", assemble(u*dx)
    v_values += [[v(p) for p in points]]
    u_values += [[u(p) for p in points]]

do_the_plot_thing = False
if do_the_plot_thing:

    import numpy
    from plot_results import *

    v_values = numpy.array(v_values)
    u_values = numpy.array(u_values)
    v_file = open("results-direct/v.txt", 'w')
    u_file = open("results-direct/u.txt", 'w')
    for i in range(v_values.shape[0]):
        v_file.write(" ".join(["%.7e" % v for v in v_values[i, :]]))
        v_file.write("\n")
        u_file.write(" ".join(["%.7e" % u for u in u_values[i, :]]))
        u_file.write("\n")

    plot_data(v_values, ylabel="v", show=False)
    plot_data(u_values, ylabel="u")

#interactive()
