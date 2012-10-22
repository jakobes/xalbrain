"""
"""

# Marie E. Rognes <meg@simula.no>
# Last changed: 2012-10-22

import math
from dolfin import *
from dolfin_adjoint import *
from beatadjoint import *

parameters["reorder_dofs"] = False
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True

class SampleConductivity(Expression):
    """Randomly chosen values by Marie to represent some spatially
    varying conductivity"""
    def eval(self, values, x):
        chi = 2000.0   # cm^{-1}
        values[0] = 3.0/chi
        #r = math.sqrt(x[0]**2 + x[1]**2)
        #if r > 0.05:
        #    values[0] = 3.0/chi
        #else:
        #    values[0] = 10.0/chi

class InitialCondition(Expression):
    def eval(self, values, x):
        values[1] = 0.0
        if (x[0] < 10):
            values[0] = 30.0
        else:
            values[0] = -85.0
    def value_shape(self):
        return (2,)

chi = 2000.0   # cm^{-1}
mesh = Mesh("../../data/meshes/mesh115_coarse_no_markers.xml.gz")
#coords = mesh.coordinates()
#print "(x_min, x_max) = ", (min(coords[:][0]), max(coords[:][0]))
#print "(y_min, y_max) = ", (min(coords[:][1]), max(coords[:][1]))
#print "(z_min, z_max) = ", (min(coords[:][2]), max(coords[:][2]))

# Woohoo:
m = SampleConductivity()
CG1 = FunctionSpace(mesh, "CG", 1)
m = Function(interpolate(m, CG1, annotate=False), name="CellConductivity")

class MyHeart(CardiacModel):
    def __init__(self, cell_model):
        CardiacModel.__init__(self, cell_model)
    def domain(self):
        return mesh
    def conductivities(self):
        # FIXME
        s_il = m
        s_it = 0.3/chi
        s_iz = 2.0/chi
        s_el = 2.0/chi
        s_et = 1.3/chi
        s_ez = 1.3/chi

        M_i = as_tensor(((s_il, 0, 0), (0, s_it, 0), (0, 0, s_iz)))
        M_e = as_tensor(((s_el, 0, 0), (0, s_et, 0), (0, 0, s_ez)))
        return (M_i, M_e)

def cell_model():
    # Set-up cell model
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
    model = FitzHughNagumo(cell_parameters)
    return model

# Setup cell and cardiac model
cell = cell_model()
heart = MyHeart(cell)

# Set-up solver
Solver = SplittingSolver
ps = Solver.default_parameters()
ps["enable_adjoint"] = True
ps["linear_variational_solver"]["linear_solver"] = "direct"
solver = Solver(heart, parameters=ps)

# Define initial condition here (no need to annotate this step)
ic0 = InitialCondition()
ic = Function(project(ic0, solver.VS, annotate=False))

# Define end-time and (constant) timestep
T = 100.0
k_n = 0.25

# Assign initial condition
(vs_, vs, u) = solver.solution_fields()
vs_.adj_name = "vs_"
vs.adj_name = "vs"
u.adj_name = "u"
vs_.assign(ic, annotate=True)

# Solve
begin("Solving primal")
solutions = solver.solve((0, T), k_n)
for (timestep, vs, u) in solutions:
    plot(u)
    continue
end()

(v, s) = split(vs)
plot(v)
interactive()

# # Define some functional
# J = Functional(inner(v, v)*ds*dt)#[FINISH_TIME])
# dJdm = compute_gradient(J, InitialConditionParameter(m))

# plot(dJdm, interactive=True, title="Sensitivity map")
