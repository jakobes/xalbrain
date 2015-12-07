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

# Define the computational domain
mesh = UnitSquareMesh(100, 100)
time = Constant(0.0)

# Define the conductivity (tensors)
M_i = 1.0
M_e = 1.0

# Define the cell model and update parameters if you wish to do so
cell_model = FitzHughNagumoManual()
cell_model_parameters = cell_model.parameters()
cell_model_parameters["a"] = Expression("0.13*(10*x[0] + 1.0)", degree=1)

# Define any external stimulus
stimulus = Expression("10*t*x[0]", t=time)

# Collect all this information into the CardiacModel class
cardiac_model = CardiacModel(mesh, time, M_i, M_e, cell_model, stimulus)

# Decide on some numerical scheme options:
ps = SplittingSolver.default_parameters()
ps["theta"] = 1.0                        # Use first order splitting
ps["CardiacODESolver"]["scheme"] = "RL1" # Use Rush-Larsen scheme

# Initialize the solver
solver = SplittingSolver(cardiac_model, params=ps)

# Extract the solution fields and set the initial conditions
(vs_, vs, vur) = solver.solution_fields()
vs_.assign(cell_model.initial_conditions())

# Solve
dt = 0.1
T = 1.0
interval = (0.0, T)
for (timestep, fields) in solver.solve(interval, dt):
    (vs_, vs, vur) = fields

# Visualize some results
plot(vs[0], title="Transmembrane potential (v) at end time")
plot(vs[1], title="State variable (s) at end time")

print vs.vector().norm("l2")

print "Success."
interactive()
