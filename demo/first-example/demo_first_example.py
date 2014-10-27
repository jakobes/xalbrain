#!/usr/bin/env python
#  -*- coding: utf-8 -*-

# .. _first_example
#
# A basic example of how to use the cbcbeat module.
#
# First example for cbcbeat
# =============================

# Import the cbcbeat module
from cbcbeat import *

# Define the cardiac model including the computational domain,
# conductivities, cell model, stimulus
mesh = UnitSquareMesh(100, 100)
time = Constant(0.0)
M_i = 1.0
M_e = 1.0
cell_models = FitzHughNagumoManual()
stimulus = {0: Expression("10*t*x[0]", t=time)}
cardiac_model = CardiacModel(mesh, time, M_i, M_e, cell_models, stimulus)

# Initialize the solver
solver = SplittingSolver(cardiac_model)

# Extract the solution fields and set the initial conditions
(vs_, vs, vur) = solver.solution_fields()
vs_.assign(cell_models.initial_conditions())

# Solve
dt = 0.1
T = 1.0
interval = (0.0, T)
for (timestep, fields) in solver.solve(interval, dt):
    (vs_, vs, vur) = fields

# Visualize some results
plot(vs[0], title="Transmembrane potential (v) at end time")
plot(vs[1], title="State variable (s) at end time")

print "Success."
interactive()
