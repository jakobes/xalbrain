#!/usr/bin/env python
#  -*- coding: utf-8 -*-

# .. _first_example
#
# A basic practical example of how to use the cbcbeat module, in
# particular how to solve the bidomain equations coupled to a
# moderately complex cell model using the splitting solver provided by
# cbcbeat.
#
# First example for cbcbeat
# =========================

# Import the cbcbeat module
from cbcbeat import *

# NB: Workaround for FEniCS 1.7.0dev
import ufl
ufl.algorithms.apply_derivatives.CONDITIONAL_WORKAROUND = True

# Turn on FFC/FEniCS optimizations
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True
flags = ["-O3", "-ffast-math", "-march=native"]
parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)
parameters["form_compiler"]["quadrature_degree"] = 3

# Turn off adjoint functionality
parameters["adjoint"]["stop_annotating"] = True

# Define the computational domain
mesh = UnitSquareMesh(100, 100)
time = Constant(0.0)

# Define the conductivity (tensors)
M_i = 2.0
M_e = 1.0

# Pick a cell model (see supported_cell_models for tested ones)
cell_model = Tentusscher_panfilov_2006_epi_cell()

# Define some external stimulus
stimulus = Expression("10*t*x[0]", t=time, degree=1)

# Collect this information into the CardiacModel class
cardiac_model = CardiacModel(mesh, time, M_i, M_e, cell_model, stimulus)

# Customize and create a splitting solver
ps = SplittingSolver.default_parameters()
ps["theta"] = 0.5                        # Second order splitting scheme
ps["pde_solver"] = "monodomain"          # Use Monodomain model for the PDEs
ps["CardiacODESolver"]["scheme"] = "RL1" # 1st order Rush-Larsen for the ODEs
ps["MonodomainSolver"]["linear_solver_type"] = "iterative"
ps["MonodomainSolver"]["algorithm"] = "cg"
ps["MonodomainSolver"]["preconditioner"] = "petsc_amg"

solver = SplittingSolver(cardiac_model, params=ps)

# Extract the solution fields and set the initial conditions
(vs_, vs, vur) = solver.solution_fields()
vs_.assign(cell_model.initial_conditions())

# Time stepping parameters
dt = 0.1
T = 1.0
interval = (0.0, T)

timer = Timer("XXX Forward solve") # Time the total solve

# Solve!
for (timestep, fields) in solver.solve(interval, dt):
    print "(t_0, t_1) = (%g, %g)", timestep

    # Extract the components of the field (vs_ at previous timestep,
    # current vs, current vur)
    (vs_, vs, vur) = fields

    # Print memory usage (just for the fun of it)
    print memory_usage()

timer.stop()

# Visualize some results
plot(vs[0], title="Transmembrane potential (v) at end time")
plot(vs[1], title="1st state variable (s_0) at end time")

# List times spent
list_timings(TimingClear_keep, [TimingType_user])

print "Success!"
interactive()
