"""
This test tests the splitting solver for the bidomain equations with a
FitzHughNagumo model.

The test case was been compared against pycc up till T = 100.0. The
relative difference in L^2(mesh) norm between beat and pycc was then
less than 0.2% for all timesteps in all variables.

The test was then shortened to T = 4.0, and the reference at that time
computed and used as a reference here.
"""

__author__ = "Marie E. Rognes (meg@simula.no), 2012--2013"
__all__ = []

import math

from dolfin import *
from beatadjoint import *
import numpy as np
import ufl

parameters["reorder_dofs_serial"] = False # Crucial because of
                                          # stimulus assumption. FIXME.
parameters["form_compiler"]["cpp_optimize"] = True
flags = ["-O3", "-ffast-math", "-march=native"]
parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)
parameters["form_compiler"]["quadrature_degree"] = 2

set_log_active(False)
set_log_level(WARNING)
ufl.set_level(WARNING)

do_plot = False

class StimSubDomain(SubDomain):
    def __init__(self, dx_stim):
        self.dx_stim = dx_stim
        SubDomain.__init__(self)
        
    def inside(self, x, on_boundary):
        return np.all(x <= self.dx_stim+DOLFIN_EPS)

def setup_model(cellmodel, domain, amplitude=50, duration=1,
                harmonic_mean=True):
    "Set-up cardiac model based on benchmark parameters."

    # Create scalar FunctionSpace
    V = FunctionSpace(domain, "CG", 1)

    # Define conductivities
    chi = 1400.0    # cm^{-1}

    s_il = 170./100/chi    # S
    s_it = 19./100./chi  # S
    s_el = 620./100/chi    # S
    s_et = 240/100/chi    # S
        
    if harmonic_mean:
        sl = s_il*s_el/(s_il+s_el)
        st = s_it*s_et/(s_it+s_et)
        M_i = as_tensor(((sl, 0, 0), (0, st, 0), (0, 0, st)))
    else:
        M_i = as_tensor(((s_il, 0, 0), (0, s_it, 0), (0, 0, s_it)))
        
    M_e = as_tensor(((s_el, 0, 0), (0, s_et, 0), (0, 0, s_et)))

    stim_marker = 1
    dx_stim = 0.15
    stim_subdomain = StimSubDomain(0.15)
    stim_domain = CellFunction("size_t", domain, 0)
    stim_subdomain.mark(stim_domain, stim_marker)
    domains = domain.domains()
    dim = domain.topology().dim()

    if do_plot:
        plot(stim_domain, interactive=True)

    # Mark domains in mesh with stim domains
    for cell in SubsetIterator(stim_domain, stim_marker):
        domains.set_marker((cell.index(), stim_marker), dim)
    
    time = Constant(0.0)
    stim = Expression("time > start ? (time <= (duration + start) ? "\
                      "amplitude : 0.0) : 0.0", time=time, duration=duration, \
                      start=0.0, amplitude=amplitude)

    heart = CardiacModel(domain, time, M_i, M_e, cellmodel, {1:stim})
    return heart

class Plotter:
    def __init__(self, up, V, cell_model_str):
        self.up = up
        self.V = V
        self.u_plot = Function(V)
        self.cell_model_str = cell_model_str

        # Setup projection system
        u, v = TrialFunction(V), TestFunction(V)
        a = u*v*dx
        self.L = up*v*dx
        self.A = assemble(a)
        
    def plot(self, interactive=False, title=""):
        b = assemble(self.L)
        solve(self.A, self.u_plot.vector(), b)
        ranges = (0., 1.0) if self.cell_model_str == "bistable" else (40., -85.)
        plot(self.u_plot, interactive=interactive, title=title, scale=0.,
             range_max=ranges[1], range_min=ranges[0])



def cell_model_parameters():
    "Set-up and return benchmark parameters for the ten Tuscher & Panfilov cell model."
    # FIXME: simon: double check that parameters are the actual benchmark parameters
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
                          ("Cm", 0.185),
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
    "Set-up and return benchmark initial conditions for the ten Tuscher & Panfilov cell model."
    ic = OrderedDict([("V", -85.23),
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
                      ("Ca_i", 0.000126),
                      ("R_prime", 0.9073),
                      ("Ca_SR", 3.64),
                      ("Ca_ss", 0.00036),
                      ("Na_i", 8.604),
                      ("K_i", 136.89)])
    return ic



def run_splitting_solver(CellModel, domain, dt, T, amplitude=50., \
                        duration=2.0, theta=1.0)

    assert CellModel == Tentusscher_panfilov_2006_epi_cell
    
    # Set-up solver
    ps = SplittingSolver.default_parameters()
    ps["pde_solver"] = "monodomain"

    ps["MonodomainSolver"]["linear_solver_type"] = "direct"
    ps["MonodomainSolver"]["theta"] = theta

    ps["theta"] = theta
    ps["enable_adjoint"] = False
    ps["apply_stimulus_current_to_pde"] = True

    cm_params = CellModel.default_parameters()
    cm_inits = CellModel.default_initial_conditions()
    cellmodel = CellModel(params=cm_params, init_conditions=cm_inits)

    heart = setup_model(cellmodel, domain, amplitude, duration,
                        harmonic_mean=ps["pde_solver"] == "monodomain")
    
    solver = SplittingSolver(heart, ps)

    (vs_, vs, vur) = solver.solution_fields()

    # FIXME: Do not hardcode this, but use something like:
    #solver.ode_solver.set_initial_conditions(v)
    # FIXME: Is this really what I want?
    vs_.vector()[:] = -85.23
    vs.vector()[:] = -85.23

    # Solve
    total = Timer("Total solver time")
    solutions = solver.solve((0, T), dt)
    plot_range = (-90., 40.)
    if do_plot:
        plot(vs_, interactive=True, title="Initial conditions", scale=0.,
             range_max=plot_range[1], range_min=plot_range[0])

    # Get the local dofs from a Function
    activation_times = Function(vs_.function_space()).vector().array()
    activation_times = 100*np.ones((activation_times.shape[0], 2))
    
    for (timestep, (v, vur)) in solutions:

        v_values = v.vector().array()
        activation_times[v_values>0, 1] = timestep[1]
        activation_times[:, 0] = activation_times.min(1)
        num_activated = (activation_times[:, 0]<100).sum()
        print "{:.2f} {:5d}/{:5d}".format(\
            timestep[1], num_activated, len(v_values))
        if do_plot:
            plot(v, title="run, t=%.3f" % timestep[1], interactive=False, scale=0.,
                 range_max=plot_range[1], range_min=plot_range[0])

        if num_activated == len(v_values):
            break
        
        continue
    
    total.stop()

    u = Function(v.function_space())
    u.vector().set_local(activation_times[:, 0].copy())

    File("activation_times_{}_{}.xdmf".format(\
        domain.num_vertices(), dt))
    activation_times[:, 0].tofile("activation_times{}_{}.np".format(\
        domain.num_vertices(), dt))

    if ps["pde_solver"] == "bidomain":
        u = project(vur[1], vur.function_space().sub(1).collapse())
    else:
        u = vur
    norm_u = norm(u)
    
    #plot(v, title="Final u, t=%.1f (%s)" % (timestep[1], ps["pde_solver"]), \
    #     interactive=True, scale=0., range_max=40., range_min=-85.)
    
if __name__ == "__main__":

    CellModel = Tentusscher_panfilov_2006_epi_cell

    stim_amplitude = 0.0
    stim_duration = 2.0

    T = 70.0  + 1.e-6  # mS 500.0

    Lx = 2.0  # cm
    Ly = 0.7  # cm
    Lz = 0.3  # cm

    theta = 0.5 # 1.0

    for dx in [0.05]:#, 0.02, 0.01]:
        for dt in [0.05]:#, 0.01, 0.005]:
            domain = BoxMesh(0.0, 0.0, 0.0,
                             Lx, Ly, Lz,
                             int(np.ceil(Lx/dx)), int(np.ceil(Ly/dx)), int(np.ceil(Lz/dx)))
            
            run_splitting_solver(CellModel, domain, \
                                dt, T, stim_amplitude, \
                                stim_duration, \
                                theta)
