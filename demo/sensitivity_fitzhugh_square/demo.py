"""
A simple demo demonstrating how to produce sensitivity maps.
"""

# Marie E. Rognes <meg@simula.no>
# Last changed: 2012-10-19

import math
from dolfin import *
from dolfin_adjoint import *
from beatadjoint import *

parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True

class SampleConductivity(Expression):
    """Randomly chosen values by Marie to represent some spatially
    varying conductivity"""
    def eval(self, values, x):
        chi = 2000.0   # cm^{-1}
        r = math.sqrt(x[0]**2 + x[1]**2)
        if r > 0.05:
            values[0] = 3.0/chi
        else:
            values[0] = 10.0/chi

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

chi = 2000.0   # cm^{-1}

mesh = UnitSquare(100, 100)
CG1 = FunctionSpace(mesh, "CG", 1)

# Woohoo:
m = SampleConductivity()
m = interpolate(m, CG1, annotate=False)

class MyHeart(CardiacModel):
    def __init__(self, cell_model):
        CardiacModel.__init__(self, cell_model)
    def domain(self):
        return mesh
    def conductivities(self):
        s_il = m
        s_it = 0.3/chi # mS
        s_el = 2.0/chi # mS
        s_et = 1.3/chi # mS
        M_i = as_tensor(((s_il, 0), (0, s_it)))
        M_e = as_tensor(((s_el, 0), (0, s_et)))
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
ps = SplittingSolver.default_parameters()
ps["enable_adjoint"] = True
ps["linear_variational_solver"]["linear_solver"] = "direct"
solver = SplittingSolver(heart, parameters=ps)

# Define initial condition here (no need to annotate this step)
ic = InitialCondition()
ic = Function(project(ic, solver.VS, annotate=False))

# Define end-time and (constant) timestep
T = 1.0
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

# Define some functional
J = Functional(inner(v, v)*ds*dt)#[FINISH_TIME])
dJdm = compute_gradient(J, InitialConditionParameter(m))

plot(dJdm, interactive=True, title="Sensitivity map")
