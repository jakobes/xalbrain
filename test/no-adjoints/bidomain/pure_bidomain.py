"""
2nd test case for bidomain solver with no cell model -- setup
suggested by Glenn in order to identify u discrepancy between beat and
pycc.
"""
# Marie E. Rognes <meg@simula.no>
# Last changed: 2012-10-10

from dolfin import *
from beatadjoint import *

parameters["reorder_dofs"] = False
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True

class InitialCondition(Expression):
    def eval(self, values, x):
        c = (0.15, 0.05)
        values[0] = 100*exp(-100*(pow((x[0]-c[0]), 2) + pow((x[1]-c[1]), 2)))
        values[1] = 0.0
    def value_shape(self):
        return (2,)

N = 100
mesh = UnitSquare(N, N)
#plot(mesh, interactive=True)
#exit()

class MyHeart(CardiacModel):
    def __init__(self, cell_model):
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
cell = NoCellModel()
heart = MyHeart(cell)

# Set-up solver
Solver = SplittingSolver
parameters = Solver.default_parameters()
parameters["enable_adjoint"] = True
parameters["theta"] = 1.0
#parameters["linear_variational_solver"]["linear_solver"] = "iterative"
parameters["linear_variational_solver"]["linear_solver"] = "direct"
solver = Solver(heart, parameters)
#theta = solver.parameters["theta"]

# Define end-time and (constant) timestep
T = 1.0 + 1.e-6
dt = 0.01
#dt = 1.0

# Define initial condition(s)
ic = InitialCondition()
vs0 = project(ic, solver.VS)
(vs_, vs, u) = solver.solution_fields()
vs_.assign(vs0)

V = solver.VS.sub(0).collapse()
v_pycc = Function(V, "pycc-data-corner/v.xml")
u_pycc = Function(V, "pycc-data-corner/u.xml")

plot(v_pycc, title="pycc v")
plot(u_pycc, title="pycc u")

# Solve
info_green("Solving primal")
solutions = solver.solve((0, T), dt)
for (timestep, vs, u) in solutions:
    (v, s) = vs.split(deepcopy=True)

print "timestep = ", timestep
plot(v, title="beat v")
plot(u, title="beat u")
vfile = File("v100.xml")
vfile << v
ufile = File("u100.xml")
ufile << u

print "||v_pycc - v||_0 = ", errornorm(v_pycc, v)
print "||u_pycc - u||_0 = ", errornorm(u_pycc, u)

interactive()




