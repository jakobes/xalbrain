"""
A first example of how to solve a basic electrophysiology problem.
"""

# Import the beatadjoint module
from beatadjoint import *

# Define the cardiac model
mesh = UnitSquareMesh(100, 100)
time = Constant(0.0)
M_i = 1.0
M_e = 1.0
cell_models = FitzHughNagumoManual()
stimulus = {0: Expression("0*t", t=time)}
cardiac_model = CardiacModel(mesh, time, M_i, M_e, cell_models, stimulus)

# Initialize the solver
solver = SplittingSolver(cardiac_model)

# Extract the solution fields and set the initial conditions
(vs_, vs, vur) = solver.solution_fields()
vs_.assign(cell_models.initial_conditions())

# Define separate function for v
v = Function(solver.VS.sub(0).collapse())

# Solve
dt = 0.1
T = 1.0
interval = (0.0, T)
for (timestep, fields) in solver.solve(interval, dt):
    (vs_, vs, vur) = fields

    # Update separate v function and plot it
    v.assign(vs.split(deepcopy=True)[0], annotate=False)
    plot(v, title="Transmembrane potential (v)")

interactive()
