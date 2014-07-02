"""
This script solves the Niederer et al 2011 benchmark
Phil.Trans. R. Soc. A 369,
"""

__author__ = "Johan Hake and Simon W. Funke (simon@simula.no), 2014"
__all__ = []

# Modified by Marie E. Rognes, 2014

from dolfin import *
from beatadjoint import *
import numpy
set_log_level(PROGRESS)

parameters["form_compiler"]["cpp_optimize"] = True
flags = ["-O3", "-ffast-math", "-march=native"]
parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)
parameters["form_compiler"]["quadrature_degree"] = 2

# MER says: should use compiled c++ expression here for vastly
# improved efficiency.
class StimSubDomain(SubDomain):
    "This represents the stimulation domain: [0, L]^3 mm."
    def __init__(self, L):
        self.L = L
        SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        return numpy.all(x <= self.L + DOLFIN_EPS)


class ActivationTimer(object):
    """ Keeps track of the pointwise activation time in the domain """
    # sf1409: needs performance improvements

    def __init__(self, V, threshold, init_value=-1.):
        """ Arguments:
          * V: A dolfin.FunctionSpace of the potential field,
          * threshold: potential threshold that defines the activation,
          * init_value: The value that should be used for points which got never activated.
        """

        self.init_value = init_value
        self.activation_time = Function(V)
        self.activation_time.vector()[:] = init_value

        self.threshold = threshold

    def update(self, timestep, potential):
        """ Arguments:
          * timestep: the current timestep,
          * V: A dolfin.Function of the potential field.
        """

        p_arr = potential.vector().array()
        at_arr = self.activation_time.vector().array()

        for i, (act_time, pot) in enumerate(zip(at_arr, p_arr)):
            # Identify points with first time activation 
            if (act_time < self.init_value + DOLFIN_EPS and 
                    pot > self.threshold + DOLFIN_EPS):
                self.activation_time.vector()[i] = timestep


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
    File("output/simulation_region.pvd") << stim_domain

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

def cell_model_initial_conditions():
    """Return initial conditions specified in the Niederer benchmark 
    for the ten Tuscher & Panfilov cell model. (Checked twice by SF and MER)"""
    ic = {"V": -85.23,       # mV
          "Xr1": 0.00621,
          "Xr2": 0.4712,
          "Xs": 0.0095,
          "m": 0.00172,
          "h": 0.7444,
          "j": 0.7045,
          "d": 3.373e-05,
          "f": 0.7888,
          "f2": 0.9755,
          "fCass": 0.9953,
          "s": 0.999998,
          "r": 2.42e-08,
          "Ca_i": 0.000126,  # millimolar
          "R_prime": 0.9073,
          "Ca_SR": 3.64,     # millimolar
          "Ca_ss": 0.00036,  # millimolar
          "Na_i": 8.604,     # millimolar
          "K_i": 136.89      # millimolar
    }
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
    cell_inits = cell_model_initial_conditions()
    cellmodel = CellModel(init_conditions=cell_inits)

    # Set-up cardiac model
    heart = setup_model(cellmodel, domain)

    # Set-up solver
    solver = SplittingSolver(heart, ps)

    # Extract the solution fields and set the initial conditions
    (vs_, vs, vur) = solver.solution_fields()
    vs_.assign(cellmodel.initial_conditions())
    solutions = solver.solve((0, T), dt)

    V = vs.function_space().sub(0).collapse()
    # Define common function for output purposes
    v_tmp = Function(V)
    activation_time_pvd = File("output/activation_time.pvd")
    v_pvd = File("output/v.pvd")

    activation_timer = ActivationTimer(V, threshold=-85.23)
    activation_time_pvd << activation_timer.activation_time

    # Solve
    total = Timer("Total beatadjoint solver time")
    for (timestep, (vs_, vs, vur)) in solutions:
        print "Solving on %s" % str(timestep)

        w = vs.split(deepcopy=True)
        activation_timer.update(timestep[0], w[0])

        v_tmp.assign(w[0], annotate=False)
        v_pvd << v_tmp
        activation_time_pvd << activation_timer.activation_time
    total.stop()

    list_timings()
    interactive()

    return activation_timer.activation_time


if __name__ == "__main__":

    T = 70.0 # mS 500.0

    # Define geometry parameters
    Lx = 20. # mm
    Ly = 7.  # mm
    Lz = 3.  # mm

    # Define solver parameters
    theta = 1.0 # 0.5

    for dx in [0.5]:# [0.5, 0.2, 0.1]:
        for dt in [0.05]:#, 0.01, 0.005]:

            # Create computational domain [0, Lx] x [0, Ly] x [0, Lz]
            # with resolution prescribed by benchmark
            N = lambda v: int(numpy.rint(v))
            domain = BoxMesh(0.0, 0.0, 0.0, Lx, Ly, Lz,
                             N(Lx/dx), N(Ly/dx), N(Lz/dx))

            # Run solver
            activation_time = run_splitting_solver(domain, dt, T, theta)

            activation_time_xml = "output/activation_time_dx=%s_dt=%s.xml" % (dx, dt)
            mesh_xml = "output/activation_time_dx=%s_dt=%s_mesh.xml" % (dx, dt)
            print "Run 'python analysis.py %s' to analyse the results." % activation_time_xml
            File(activation_time_xml) << activation_time
            File(mesh_xml) << domain
