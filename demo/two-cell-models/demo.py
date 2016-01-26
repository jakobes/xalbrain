"""
This example acts is benchmark tuned for computational efficiency
for a monodomain + moderately complex (ten Tusscher) cell model
solver.
"""

__author__ = "Marie E Rognes, Johan Hake and Patrick Farrell"

import numpy
import sys

from cbcbeat import *

# Set FFC some parameters
#parameters["form_compiler"]["cpp_optimize"] = True
#flags = ["-O3", "-ffast-math", "-march=native"]
#parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)
#parameters["form_compiler"]["quadrature_degree"] = 3

# Customize petsc_amg (GAMG) parameters via the oh-so-readable petsc
# command-line interface
# args = [sys.argv[0]] + """
#                        --petsc.ksp_monitor_cancel
#                        --petsc.ksp_monitor
#                        --petsc.ksp_converged_reason
#                        --petsc.ksp_type cg
#                        --petsc.pc_type gamg
#                        --petsc.pc_gamg_verbose 10
#                        --petsc.pc_gamg_square_graph 0
#                        --petsc.pc_gamg_coarse_eq_limit 3000
#                        --petsc.mg_coarse_pc_type redundant
#                        --petsc.mg_coarse_sub_pc_type lu
#                        --petsc.mg_levels_ksp_type richardson
#                        --petsc.mg_levels_ksp_max_it 3
#                        --petsc.mg_levels_pc_type sor
#                        """.split()
# parameters.parse(argv=args)


parameters["adjoint"]["stop_annotating"] = True

# Define space and time
#mesh = UnitIntervalMesh(4)
n = 4
mesh = UnitSquareMesh(n, n)
time = Constant(0.0)

# Surface to volume ratio and membrane capacitance
chi = 140.0     # mm^{-1}
C_m = 0.01      # mu F / mm^2

# Define conductivity tensor
M_i = 1.0
M_e = 1.0

# Define two different cell models on the mesh
c0 = Beeler_reuter_1977()
#c1 = Beeler_reuter_1977()
#c1 = FitzHughNagumoManual()
c1 = Fenton_karma_1998_BR_altered()
markers = CellFunction("uint", mesh, 0)
markers.array()[::2] = 2
cell_model = MultiCellModel((c0, c1), (2, 0), markers)

solver = BasicCardiacODESolver(mesh, time, cell_model,
                               I_s=Expression("1000*x[0]*t", t=time),
                               params=None)
dt = 0.01
T = 10*dt

# Assign initial conditions
(vs_, vs) = solver.solution_fields()
ic = cell_model.initial_conditions()
vs_.assign(cell_model.initial_conditions())
vs.assign(vs_)

solutions = solver.solve((0.0, T), dt)

V = vs.split()[0].function_space().collapse()
v = Function(V)
for ((t0, t1), y) in solutions:
    v.assign(y.split(deepcopy=True)[0])
    plot(v)

interactive()

# #cellmodels = Tentusscher_panfilov_2006_epi_cell()

# # Store input parameters in cardiac model
# I_s = Constant(1.0)

# heart = CardiacModel(mesh, time, M_i, M_e, cellmodels, I_s)

# # Customize and create monodomain solver
# ps = SplittingSolver.default_parameters()
# ps["pde_solver"] = "monodomain"
# ps["apply_stimulus_current_to_pde"] = True

# # Create solver
# solver = SplittingSolver(heart, params=ps)

# # Extract the solution fields and set the initial conditions
# dt = 0.01
# T = 10*dt
# (vs_, vs, vur) = solver.solution_fields()
# vs_.assign(cellmodels.initial_conditions())
# solutions = solver.solve((0, T), dt)

# # Solve
# total = Timer("XXX Total cbcbeat solver time")
# for (timestep, (vs_, vs, vur)) in solutions:
#     print "Solving on %s" % str(timestep)

#     # Print memory usage (just for the fun of it)
#     print memory_usage()

# total.stop()

# # Plot result (as sanity check)
# #plot(vs[0], interactive=True)

# # Stop timer and list timings
# if MPI.rank(mpi_comm_world()) == 0:
#     list_timings(TimingClear_keep, [TimingType_wall])
