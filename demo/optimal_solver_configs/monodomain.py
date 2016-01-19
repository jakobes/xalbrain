"""
This example acts is benchmark tuned for computational efficiency
for a monodomain + moderately complex (ten Tusscher) cell model
solver.
"""

__author__ = "Marie E Rognes, Johan Hake and Patrick Farrell"

import numpy
import sys

from cbcbeat import *

# Set FFC parameters
parameters["form_compiler"]["cpp_optimize"] = True
flags = ["-O3", "-ffast-math", "-march=native"]
parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)
parameters["form_compiler"]["quadrature_degree"] = 3

args = [sys.argv[0]] + """
                       --petsc.ksp_monitor_cancel
                       --petsc.ksp_monitor
                       --petsc.ksp_converged_reason
                       --petsc.ksp_type cg
                       --petsc.pc_type gamg
                       --petsc.pc_gamg_verbose 10
                       --petsc.pc_gamg_square_graph 0
                       --petsc.pc_gamg_coarse_eq_limit 3000
                       --petsc.mg_coarse_pc_type redundant
                       --petsc.mg_coarse_sub_pc_type lu
                       --petsc.mg_levels_ksp_type richardson
                       --petsc.mg_levels_ksp_max_it 3
                       --petsc.mg_levels_pc_type sor
                       """.split()
#parameters.parse(argv=args)

# MER says: should use compiled c++ expression here for vastly
# improved efficiency.
class StimSubDomain(SubDomain):
    "This represents the stimulation domain: [0, L]^3 mm."
    def __init__(self, L):
        self.L = L
        SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        return numpy.all(x <= self.L + DOLFIN_EPS)

def define_conductivity_tensor(chi, C_m):
    "Just define the conductivity tensor"

    # Realistic conductivities
    sigma_il = 0.17  # mS/mm
    sigma_it = 0.019 # mS/mm
    sigma_el = 0.62  # mS/mm
    sigma_et = 0.24  # mS/mm

    # Compute monodomain approximation by taking harmonic mean in each
    # direction of intracellular and extracellular part
    def harmonic_mean(a, b):
        return a*b/(a + b)

    sigma_l = harmonic_mean(sigma_il, sigma_el)
    sigma_t = harmonic_mean(sigma_it, sigma_et)

    # Scale conducitivites by 1/(C_m * chi)
    s_l = sigma_l/(C_m*chi) # mm^2 / ms
    s_t = sigma_t/(C_m*chi) # mm^2 / ms

    # Define conductivity tensor
    M = as_tensor(((s_l, 0, 0), (0, s_t, 0), (0, 0, s_t)))
    return M

def setup_model(cellmodel, domain):
    "Set-up cardiac model based on benchmark parameters."

    # Define time
    time = Constant(0.0)

    # Surface to volume ratio and membrane capacitance
    chi = 140.0     # mm^{-1}
    C_m = 0.01      # mu F / mm^2

    # Define conductivity tensor
    M = define_conductivity_tensor(chi, C_m)

    # Define stimulation region defined as [0, L]^3
    stimulus_domain_marker = 1
    L = 1.5
    stimulus_subdomain = StimSubDomain(L)
    markers = CellFunction("size_t", domain, 0)
    markers.set_all(0)
    stimulus_subdomain.mark(markers, stimulus_domain_marker)

    # Define stimulation protocol
    stimulation_protocol_duration = 2.0 # ms
    A = 50000.0                         # mu A/cm^3
    cm2mm = 10.0
    stimulation_protocol_amplitude = A/(chi*C_m)*(1./cm2mm)**3 # mV/ms
    stim = Expression("time >= start ? (time <= (duration + start) ? amplitude : 0.0) : 0.0",
                      time=time,
                      start=0.0,
                      duration=stimulation_protocol_duration,
                      amplitude=stimulation_protocol_amplitude)

    # Store input parameters in cardiac model
    I_s = Markerwise((stim,), (1,), markers)
    heart = CardiacModel(domain, time, M, None, cellmodel, I_s)
    return heart

def run_splitting_solver(domain, dt, T):

    # Create cardiac model  problem description
    cell_model = Tentusscher_panfilov_2006_epi_cell()
    heart = setup_model(cell_model, domain)

    # Customize and create monodomain solver
    ps = SplittingSolver.default_parameters()
    ps["pde_solver"] = "monodomain"
    ps["apply_stimulus_current_to_pde"] = True

    # 2nd order splitting scheme
    ps["theta"] = 0.5

    # Use explicit first-order Rush-Larsen scheme for the ODEs
    ps["ode_solver_choice"] = "CardiacODESolver"
    ps["CardiacODESolver"]["scheme"] = "RL1"

    # Crank-Nicolson discretization for PDEs in time:
    ps["MonodomainSolver"]["theta"] = 0.5
    ps["MonodomainSolver"]["linear_solver_type"] = "iterative"
    ps["MonodomainSolver"]["algorithm"] = "cg"
    ps["MonodomainSolver"]["preconditioner"] = "petsc_amg"

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
    plot(vs[0], interactive=True)

    # Stop timer and list timings
    if MPI.rank(mpi_comm_world()) == 0:
        list_timings(TimingClear_keep, [TimingType_wall])


if __name__ == "__main__":

    parameters["adjoint"]["stop_annotating"] = True

    # Define geometry parameters (in mm)
    Lx = 20.0; Ly = 7.0; Lz = 3.0  # mm

    # Define discretization resolutions
    dx = 0.1
    dt = 0.01

    T = 10*dt

    N = lambda v: int(numpy.rint(v))
    x0 = Point(numpy.array((0.0, 0.0, 0.0)))
    x1 = Point(numpy.array((Lx, Ly, Lz)))
    mesh = BoxMesh(x0, x1, N(Lx/dx), N(Ly/dx), N(Lz/dx))

    run_splitting_solver(mesh, dt, T)
