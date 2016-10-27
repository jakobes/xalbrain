"""
This example acts is benchmark tuned for computational efficiency
for a bidomain + moderately complex (ten Tusscher) cell model
solver.
"""

__author__ = "Marie E Rognes, Johan Hake and Patrick Farrell"

import numpy
import sys

from cbcbeat import *
from monodomain import StimSubDomain, define_conductivity_tensor
from monodomain import setup_model

set_log_level(PROGRESS)

# Set FFC some parameters
parameters["form_compiler"]["cpp_optimize"] = True
flags = ["-O3", "-ffast-math", "-march=native"]
parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)
parameters["form_compiler"]["quadrature_degree"] = 3

def run_splitting_solver(domain, dt, T):

    # Create cardiac model  problem description
    cell_model = Tentusscher_panfilov_2006_epi_cell()
    heart = setup_model(cell_model, domain)

    # Customize and create monodomain solver
    ps = SplittingSolver.default_parameters()
    ps["pde_solver"] = "bidomain"
    ps["apply_stimulus_current_to_pde"] = True

    # 2nd order splitting scheme
    ps["theta"] = 0.5

    # Use explicit first-order Rush-Larsen scheme for the ODEs
    ps["ode_solver_choice"] = "CardiacODESolver"
    ps["CardiacODESolver"]["scheme"] = "RL1"

    # Crank-Nicolson discretization for PDEs in time:
    ps["BidomainSolver"]["theta"] = 0.5
    ps["BidomainSolver"]["linear_solver_type"] = "iterative"
    ps["BidomainSolver"]["algorithm"] = "cg"
    ps["BidomainSolver"]["preconditioner"] = "fieldsplit"

    # Create solver
    solver = SplittingSolver(heart, params=ps)

    # Extract the solution fields and set the initial conditions
    (vs_, vs, vur) = solver.solution_fields()
    vs_.assign(cell_model.initial_conditions())
    solutions = solver.solve((0, T), dt)

    # Solve
    total = Timer("XXX Total cbcbeat solver time")
    for (timestep, (vs_, vs, vur)) in solutions:
        print "Solving on %s" % str(timestep)

        # Print memory usage (just for the fun of it)
        print memory_usage()

    total.stop()

    # Plot result (as sanity check)
    #plot(vs[0], interactive=True)

    # Stop timer and list timings
    if MPI.rank(mpi_comm_world()) == 0:
        list_timings(TimingClear_keep, [TimingType_wall])


if __name__ == "__main__":

    parameters["adjoint"]["stop_annotating"] = True

    # Define geometry parameters (in mm)
    Lx = 20.0; Ly = 7.0; Lz = 3.0  # mm

    # Define discretization resolutions
    dx = 0.2
    dt = 0.01

    T = 10*dt

    N = lambda v: int(numpy.rint(v))
    x0 = Point(numpy.array((0.0, 0.0, 0.0)))
    x1 = Point(numpy.array((Lx, Ly, Lz)))
    mesh = BoxMesh(x0, x1, N(Lx/dx), N(Ly/dx), N(Lz/dx))
    run_splitting_solver(mesh, dt, T)
