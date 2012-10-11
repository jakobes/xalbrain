"""
2nd test case for bidomain solver with no cell model -- setup
suggested by Glenn in order to identify u discrepancy between beat and
pycc.
"""
# Marie E. Rognes <meg@simula.no>
# Last changed: 2012-10-10

import sys
from dolfin import *
from beatadjoint import *

parameters["reorder_dofs"] = False
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True

assert (len(sys.argv) > 1), "Expecting one argument: 0 (no R) or > 0 (use R)"
use_r = int(sys.argv[1]) > 0
if use_r:
    info_green("Using R")
else:
    info_green("NOT using R")

class InitialCondition(Expression):
    def eval(self, values, x):
        values[0] = 100*exp(-100*(pow((x[0]-0.5), 2) + pow((x[1]-0.5), 2)))
        values[1] = 0.0 # u
        if use_r:
            values[2] = 0.0 # r
    def value_shape(self):
        if use_r:
            return (3,)
        else:
            return (2,)

N = 100
mesh = UnitSquare(N, N)
class MyHeart(CardiacModel):
    def __init__(self, cell_model=None):
        CardiacModel.__init__(self, cell_model)
    def domain(self):
        return mesh
    def conductivities(self):
        chi = 2000.0   # cm^{-1}
        s_il = 3.0/chi # mS
        s_it = 0.3/chi # mS
        s_el = 2.0/chi # mS
        s_et = 1.3/chi # mS
        M_i = as_tensor(((s_il, 0), (0, s_it)))
        M_e = as_tensor(((s_el, 0), (0, s_et)))
        return (M_i, M_e)

# Set-up model
heart = MyHeart()

# Set-up solver
ps = CoupledBidomainSolver.default_parameters()
ps["theta"] = 1.0
ps["real_constraint"] = use_r
#ps["linear_variational_solver"]["linear_solver"] = "iterative"
solver = CoupledBidomainSolver(heart, ps)

# Define end-time and (constant) timestep
T = 1.0 + 1.e-6
dt = 0.01

# Define initial condition(s)
ic = InitialCondition()
w0 = project(ic, solver.W)
(w_, w) = solver.solution_fields()
w_.assign(w0)

set_log_level(PROGRESS)

# Solve
info_green("Solving primal")
solutions = solver.solve((0, T), dt)
v_plot = Function(solver.W.sub(0).collapse())
u_plot = Function(solver.W.sub(1).collapse())

v_pycc = Function(V, "pycc-dat-corner/v.xml")
u_pycc = Function(V, "pycc-dat-corner/u.xml")

#if use_r:
#    files = File("pure_bidomain_r_comparison_data/u_r_krylov.pvd")
#else:
#    files = File("pure_bidomain_r_comparison_data/u_normalized_krylov.pvd")

for (timestep, w) in solutions:
    fields = w.split()

    v_plot.assign(fields[0], annotate=False)
    u_plot.assign(fields[1], annotate=False)

plot(v_plot, title="v")
plot(u_plot, title="u")
interactive()


# 0: no R iterative
# 1: with R iterative
# 2: with R direct
