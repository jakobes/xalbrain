#!/usr/bin/env python
#  -*- coding: utf-8 -*-

# .. _second_example
#
# Second example for cbcbeat
# =============================

# Import the cbcbeat module
from cbcbeat import *

# Set optimization parameters
parameters["form_compiler"]["cpp_optimize"] = True
flags = ["-O3", "-ffast-math", "-march=native"]
parameters["form_compiler"]["cpp_optimize_flags"]= " ".join(flags)
parameters["form_compiler"]["quadrature_degree"] = 4

# Print list of supported models
for model in supported_cell_models:
    print model

# Define the computational domain
mesh = UnitSquareMesh(100, 100)
time = Constant(0.0)

# Define the conductivity (tensors)
M_i = 1.0
M_e = 1.0

# Define the cell model and update parameters if you wish to do so
#cell_model = Tentusscher_panfilov_2006_epi_cell()
cell_model = Grandi_pasqualini_bers_2010()

# Define any external stimulus
stimulus = Expression("10*t*x[0]", t=time)

# Collect all this information into the CardiacModel class
cardiac_model = CardiacModel(mesh, time, M_i, M_e, cell_model, stimulus)

# Decide on some numerical scheme options:
ps = SplittingSolver.default_parameters()
#ps["theta"] = 1.0                        # Use first order splitting
#ps["CardiacODESolver"]["scheme"] = "RL1" # Use Rush-Larsen scheme

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
plot(vs[1], title="One state variable (s0) at end time")

print vs.vector().norm("l2")

print "Success."
interactive()
