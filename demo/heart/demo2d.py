"""
"""

# Marie E. Rognes <meg@simula.no>
# Last changed: 2012-10-23

import math
from dolfin import *
from dolfin_adjoint import *
from beatadjoint import *

#parameters["reorder_dofs"] = False
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = False

set_log_level(PROGRESS)

# FIXMEs:
chi = 2000.0   # Membrane surface-to-volume ratio (Need value + unit)
C_m = 1.0      # Membrane capacitance per unit area (Need value + unit)
T = 5.0        # End time (need value + unit)
k_n = 0.25     # Timestep (need value)

factor = 10.

# FIXME: Define some 'stimulation protocol': Johan will fix
class Pulse(Expression):
    def eval(self, values, x):
        values[0] = 0.0
        if (x[0] > 0.9):
            values[0] = factor*1.0

# FIXME: Get finer mesh and fibers from Molly/Sjur/Johan
#mesh = Mesh("../../data/meshes/mesh115_coarse_no_markers.xml.gz")
mesh = UnitSquare(50, 50)

# FIXME: All of the conductivities need specification in healthy and
# ischemic tissue (Johan and Molly will give to marie Marie)
class SampleConductivity(Expression):
    """Randomly chosen values by Marie to represent some spatially
    varying conductivity"""
    def eval(self, values, x):
        origo = (0.8, 0.5)
        r = 0.1
        if ((x[0] - origo[0])**2 + (x[1] - origo[1])**2 < r**2):
            values[0] = factor*3.0/(chi*C_m)
        else:
            values[0] = factor**2*3.0/(chi*C_m)

# This will be changed
m = SampleConductivity()
CG1 = FunctionSpace(mesh, "CG", 1)
m = Function(interpolate(m, CG1, annotate=False), name="CellConductivity")
plot(m, interactive=True)

class MyHeart(CardiacModel):
    def __init__(self, cell_model):
        CardiacModel.__init__(self, cell_model)
    def domain(self):
        return mesh
    def conductivities(self):
        # FIXME cf comments above
        s_il = m
        s_it = factor*0.3/(chi*C_m)
        s_el = factor*2.0/(chi*C_m)
        s_et = factor*1.3/(chi*C_m)

        M_i = as_tensor(((s_il, 0), (0, s_it)))
        M_e = as_tensor(((s_el, 0), (0, s_et)))

        return (M_i, M_e)

# Setup cell and cardiac model
cell = OriginalFitzHughNagumo()
heart = MyHeart(cell)

# Add given current as Stimulus
heart.stimulus = Pulse()

# Set-up solver
Solver = SplittingSolver
ps = Solver.default_parameters()
ps["enable_adjoint"] = True
ps["linear_variational_solver"]["linear_solver"] = "direct"
solver = Solver(heart, parameters=ps)

# Define initial condition here (no need to annotate this step)
ic = project(cell.initial_conditions(), solver.VS, annotate=False)

# Assign initial condition
(vs_, vs, u) = solver.solution_fields()
vs_.adj_name = "vs_"
vs.adj_name = "vs"
u.adj_name = "u"
vs_.assign(ic, annotate=True)

# Solve
begin("Solving primal")
solutions = solver.solve((0, T), k_n)
file = File("results-2d/u.pvd")
v_plot = Function(solver.VS.sub(0).collapse())
for (timestep, vs, u) in solutions:
    v_plot.assign(vs.split(deepcopy=True)[0], annotate=False)
    plot(v_plot, title="Membrane potential (v)")
    #plot(u, title="Extracellular potential (u)")
    #file << u
end()

(v, s) = split(vs)
plot(v_plot, title="Transmembrane potential (v)")
interactive()

# # Define some functional
J = Functional(inner(v, v)*ds*dt)
dJdm = compute_gradient(J, InitialConditionParameter(m))
plot(dJdm, interactive=True, title="Adjoint sensitivity map")
