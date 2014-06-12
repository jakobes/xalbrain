"""
This script solves the Niederer et al 2011 benchmark
Phil.Trans. R. Soc. A 369,
"""

__author__ = "Johan Hake and Simon W. Funke (simon@simula.no), 2014"
__all__ = []

# Modified by Marie E. Rognes, 2014

import math

# FIXME: Is this user-friendly? (Low priority atm)
from collections import OrderedDict

from dolfin import *
from beatadjoint import *
import numpy
import ufl

parameters["form_compiler"]["cpp_optimize"] = True
flags = ["-O3", "-ffast-math", "-march=native"]
parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)
parameters["form_compiler"]["quadrature_degree"] = 2

set_log_active(False)
set_log_level(WARNING)
ufl.set_level(WARNING)

do_plot = True

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

    # Conductivities as defined by page 4339 of Niederer benchmark
    sigma_il = 0.17  # mS / mm
    sigma_it = 0.019 # mS / mm
    sigma_el = 0.62  # mS / mm
    sigma_et = 0.24  # mS / mm

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
    """Set-up cardiac model based on benchmark parameters.

    * domain is the spatial domain/mesh
    """

    # Define time
    time = Constant(0.0)

    # Surface to volume ratio
    chi = 140.0     # mm^{-1}
    # Membrane capacitance
    C_m = 0.01 # mu F / mm^2

    # Define conductivity tensor
    M = define_conductivity_tensor(chi, C_m)

    # Define stimulation region defined as [0, L]^3
    stim_marker = 1
    L = 1.5
    stim_subdomain = StimSubDomain(L)
    stim_domain = CellFunction("size_t", domain, 0)
    stim_domain.set_all(0)
    stim_subdomain.mark(stim_domain, stim_marker)
    if do_plot:
        plot(stim_domain, title="Stimulation region")

    # FIXME: MER: We are NOT going to attach domains to the
    # mesh. Figure out a way to expose the right functionality. A
    # possibly better way of doing this is to pass the mesh function
    # into the solver(s).
    # Mark domains in mesh with stim domains
    domains = domain.domains()
    dim = domain.topology().dim()
    for cell in SubsetIterator(stim_domain, stim_marker):
        domains.set_marker((cell.index(), stim_marker), dim)

    # Define stimulation (NB: region of interest carried by the mesh
    # and assumptions in beatadjoint)
    stimulation_protocol_duration = 2. # ms
    A = 50000. # mu A/cm^3
    cm2mm = 10.
    factor = 1.0/(chi*C_m) # NB: beatadjoint convention
    stimulation_protocol_amplitude = factor*A*(1./cm2mm)**3 # mV/ms
    stim = Expression("time >= start ? (time <= (duration + start) ? amplitude : 0.0) : 0.0",
                      time=time,
                      start=0.0,
                      duration=stimulation_protocol_duration,
                      amplitude=stimulation_protocol_amplitude)

    # Store input parameters in cardiac model
    heart = CardiacModel(domain, time, M, None, cellmodel, {1:stim})

    return heart

def cell_model_parameters():
    """Set-up and return benchmark parameters for the ten Tuscher & Panfilov cell model."""
    # FIXME: simon: double check that parameters are the actual
    # benchmark parameters
    params = OrderedDict([("P_kna", 0.03),
                          ("g_K1", 5.405),
                          ("g_Kr", 0.153),
                          ("g_Ks", 0.392),
                          ("g_Na", 14.838),
                          ("g_bna", 0.00029),
                          ("g_CaL", 3.98e-05),
                          ("g_bca", 0.000592),
                          ("g_to", 0.294),
                          ("K_mNa", 40),
                          ("K_mk", 1),
                          ("P_NaK", 2.724),
                          ("K_NaCa", 1000),
                          ("K_sat", 0.1),
                          ("Km_Ca", 1.38),
                          ("Km_Nai", 87.5),
                          ("alpha", 2.5),
                          ("gamma", 0.35),
                          ("K_pCa", 0.0005),
                          ("g_pCa", 0.1238),
                          ("g_pK", 0.0146),
                          ("Buf_c", 0.2),
                          ("Buf_sr", 10),
                          ("Buf_ss", 0.4),
                          ("Ca_o", 2),
                          ("EC", 1.5),
                          ("K_buf_c", 0.001),
                          ("K_buf_sr", 0.3),
                          ("K_buf_ss", 0.00025),
                          ("K_up", 0.00025),
                          ("V_leak", 0.00036),
                          ("V_rel", 0.102),
                          ("V_sr", 0.001094),
                          ("V_ss", 5.468e-05),
                          ("V_xfer", 0.0038),
                          ("Vmax_up", 0.006375),
                          ("k1_prime", 0.15),
                          ("k2_prime", 0.045),
                          ("k3", 0.06),
                          ("k4", 0.005),
                          ("max_sr", 2.5),
                          ("min_sr", 1),
                          ("Na_o", 140),
                          ("Cm", 0.185), # FIXME: Consistency of this?!
                          ("F", 96485.3415),
                          ("R", 8314.472),
                          ("T", 310),
                          ("V_c", 0.016404),
                          ("stim_amplitude", 0),
                          ("stim_duration", 1),
                          ("stim_period", 1000),
                          ("stim_start", 10),
                          ("K_o", 5.4)])
    return params

def cell_model_initial_conditions():
    """Set-up and return benchmark initial conditions for the ten
    Tuscher & Panfilov cell model. (Checked twice by SF and MER) """
    ic = OrderedDict([("V", -85.23),  # mV
                      ("Xr1", 0.00621),
                      ("Xr2", 0.4712),
                      ("Xs", 0.0095),
                      ("m", 0.00172),
                      ("h", 0.7444),
                      ("j", 0.7045),
                      ("d", 3.373e-05),
                      ("f", 0.7888),
                      ("f2", 0.9755),
                      ("fCass", 0.9953),
                      ("s", 0.999998),
                      ("r", 2.42e-08),
                      ("Ca_i", 0.000126), # millimolar
                      ("R_prime", 0.9073),
                      ("Ca_SR", 3.64),    # millimolar
                      ("Ca_ss", 0.00036), # millimolar
                      ("Na_i", 8.604),    # millimolar
                      ("K_i", 136.89)])   # millimolar
    return ic

def run_splitting_solver(domain, dt, T, theta=1.0):

    # cell model defined by benchmark specifications
    CellModel = Tentusscher_panfilov_2006_epi_cell

    # Set-up solver
    ps = SplittingSolver.default_parameters()
    ps["pde_solver"] = "monodomain"
    ps["MonodomainSolver"]["linear_solver_type"] = "direct"
    ps["MonodomainSolver"]["theta"] = theta

    ps["theta"] = theta
    ps["enable_adjoint"] = False
    ps["apply_stimulus_current_to_pde"] = True

    # Customize cell model parameters based on benchmark specifications
    cell_params = cell_model_parameters()
    cell_inits = cell_model_initial_conditions()
    cellmodel = CellModel(params=cell_params, init_conditions=cell_inits)

    # Set-up cardiac model
    heart = setup_model(cellmodel, domain)

    # Set-up solver
    solver = SplittingSolver(heart, ps)

    # Extract the solution fields and set the initial conditions
    (vs_, vs, vur) = solver.solution_fields()
    vs_.assign(cellmodel.initial_conditions())

    # Solve
    total = Timer("Total solver time")

    solutions = solver.solve((0, T), dt)

    v = Function(vs.function_space().sub(0).collapse())

    for (timestep, (vs_, vs, vur)) in solutions:
        print "Solving on %s" % str(timestep)
        if do_plot:
            plot_range = (-90., 40.)
            w = vs.split(deepcopy=True)
            v.assign(w[0], annotate=False)
            plot(v, title="v")

    interactive()
    exit()

    # # Get the local dofs from a Function
    # activation_times = Function(vs_.function_space()).vector().array()
    # activation_times = 100*numpy.ones((activation_times.shape[0], 2))

    # for (timestep, (vs_, vs, vur)) in solutions:

    #     vs_values = vs.vector().array()
    #     activation_times[vs_values>0, 1] = timestep[1]
    #     activation_times[:, 0] = activation_times.min(1)
    #     num_activated = (activation_times[:, 0]<100).sum()
    #     print "{:.2f} {:5d}/{:5d}".format(\
    #         timestep[1], num_activated, len(vs_values))
    #     if do_plot:
    #         plot(vs, title="run, t=%.3f" % timestep[1], interactive=False, scale=0.,
    #              range_max=plot_range[1], range_min=plot_range[0])

    #     if num_activated == len(vs_values):
    #         break

    #     continue

    # total.stop()

    # u = Function(vs_.function_space())
    # u.vector().set_local(activation_times[:, 0].copy())

    # File("activation_times_{}_{}.xdmf".format(\
    #     domain.num_vertices(), dt))
    # activation_times[:, 0].tofile("activation_times{}_{}.numpy".format(\
    #     domain.num_vertices(), dt))

    # if ps["pde_solver"] == "bidomain":
    #     u = project(vur[1], vur.function_space().sub(1).collapse())
    # else:
    #     u = vur
    # norm_u = norm(u)

    #plot(v, title="Final u, t=%.1f (%s)" % (timestep[1], ps["pde_solver"]), \
    #     interactive=True, scale=0., range_max=40., range_min=-85.)

if __name__ == "__main__":

    #T = 70.0 + 1.e-6  # mS 500.0
    T = 0.2  # FIXME: Reduced time only for testing purposes

    # Define geometry parameters
    Lx = 20. # mm
    Ly = 7.  # mm
    Lz = 3.  # mm

    # Define solver parameters
    theta = 1.0 # 0.5

    for dx in [0.5]:#, 0.2, 0.1]:
        for dt in [0.05]:#, 0.01, 0.005]:

            # Create computational domain [0, Lx] x [0, Ly] x [0, Lz]
            # with resolution prescribed by benchmark
            N = lambda v: int(numpy.rint(v))
            domain = BoxMesh(0.0, 0.0, 0.0, Lx, Ly, Lz,
                             N(Lx/dx), N(Ly/dx), N(Lz/dx))

            # Run solver
            run_splitting_solver(domain, dt, T, theta)
