"""
Demo for propagation of electric potential through left and right
ventricles.
"""

# Marie E. Rognes <meg@simula.no>
# Last changed: 2012-10-27

import math
from dolfin import *
from dolfin_adjoint import *
from beatadjoint import *
import time

set_log_level(PROGRESS)

# Setup application parameters and parse from command-line
application_parameters = Parameters("Application")
application_parameters.add("T", 100.0)      # End time  (ms)
application_parameters.add("timestep", 1.0) # Time step (ms)
application_parameters.add("directory", "default-adjoint-results")
application_parameters.add("backend", "PETSc")
application_parameters.add("stimulus_amplitude", 30.0)
application_parameters.parse()
info(application_parameters, True)

# Update backend from application parameters
parameters["linear_algebra_backend"] = application_parameters["backend"]

# Adjust some general parameters
parameters["reorder_dofs_serial"] = False # Crucial!
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True

# Generic cardiac parameters
#chi = 1400.0   # Membrane surface-to-volume ratio (1/cm)
# NB : If not 1 => must scale ionic current! see book p. 55
#C_m = 1.0      # Membrane capacitance per unit area (micro F/(cm^2))

# Initialize domain
mesh = Mesh("data/mesh115_refined.xml.gz")
mesh.coordinates()[:] /= 1000.0 # Scale mesh from micrometer to millimeter
mesh.coordinates()[:] /= 10.0   # Scale mesh from millimeter to centimeter
mesh.coordinates()[:] /= 4.0    # Scale mesh as indicated by Johan/Molly

# Extract time and time-step
T = application_parameters["T"]
k_n = application_parameters["timestep"]

# Load fibers and sheets
Vv = VectorFunctionSpace(mesh, "DG", 0)
fiber = Function(Vv)
File("data/fibers.xml.gz") >> fiber
sheet = Function(Vv)
File("data/sheet.xml.gz") >> sheet
cross_sheet = Function(Vv)
File("data/cross_sheet.xml.gz") >> cross_sheet

# Extract conductivity data
V = FunctionSpace(mesh, "CG", 1)
g_el_var = Function(V, "data/g_el_field.xml.gz", name="g_el_var")
g_et_var = Function(V, "data/g_et_field.xml.gz", name="g_et_var")
g_il_var = Function(V, "data/g_il_field.xml.gz", name="g_il_var")
g_it_var = Function(V, "data/g_it_field.xml.gz", name="g_it_var")

# A touch of dolfin-adjoint magic, hopefully an optimization.
g_el_field = Function(V, name="g_el_field")
g_el_field.assign(g_el_var, annotate=True, force=True)
g_et_field = Function(V, name="g_et_field")
g_et_field.assign(g_et_var, annotate=True, force=True)
g_il_field = Function(V, name="g_il_field")
g_il_field.assign(g_il_var, annotate=True, force=True)
g_it_field = Function(V, name="g_it_field")
g_it_field.assign(g_it_var, annotate=True, force=True)

# Construct conductivity tensors from directions and conductivity
# values relative to that coordinate system
A = as_matrix([[fiber[0], sheet[0], cross_sheet[0]],
               [fiber[1], sheet[1], cross_sheet[1]],
               [fiber[2], sheet[2], cross_sheet[2]]])
M_e_star = diag(as_vector([g_el_field, g_et_field, g_et_field]))
M_i_star = diag(as_vector([g_il_field, g_it_field, g_it_field]))
M_e = A*M_e_star*A.T
M_i = A*M_i_star*A.T

# Model of the whole heart given a cell-model, using the above domain
# and conductivities
class MyHeart(CardiacModel):
    def __init__(self, cell_model):
        CardiacModel.__init__(self, cell_model)
    def domain(self):
        return mesh
    def conductivities(self):
        return (M_i, M_e)

# Setup cell model based on parameters from Glenn, which seems to be a
# little more excitable than the default FitzHugh-Nagumo parameters
# from the book.
k = 0.00004; Vrest = -85.; Vthreshold = -70.;
Vpeak = 40.; k = 0.00004; l = 0.63; b = 0.013; v_amp = Vpeak - Vrest
cell_parameters = {"c_1": k*v_amp**2, "c_2": k*v_amp, "c_3": b/l,
                   "a": (Vthreshold - Vrest)/v_amp, "b": l,
                   "v_rest":Vrest, "v_peak": Vpeak}
cell = OriginalFitzHughNagumo(cell_parameters)
heart = MyHeart(cell)

# Define some simulation protocol (use cpp expression for speed)
stimulation_cells = MeshFunction("uint", mesh, "data/stimulation_cells.xml.gz")
from stimulation import cpp_stimulus
pulse = Expression(cpp_stimulus)
pulse.cell_data = stimulation_cells
amp = application_parameters["stimulus_amplitude"]
pulse.amplitude = amp #
pulse.duration = 10.0 # ms
pulse.t = 0.0         # ms

heart.stimulus = pulse

# Set-up solver
begin("Setting-up solver")
Solver = SplittingSolver
ps = Solver.default_parameters()
ps["default_timestep"] = k_n
ps["enable_adjoint"] = True
ps["linear_variational_solver"]["linear_solver"] = "direct"
solver = Solver(heart, parameters=ps)
end()

# Define initial condition here (no need to annotate this step)
begin("Projecting initial condition")
ic = project(cell.initial_conditions(), solver.VS, annotate=False)
end()

# Assign initial condition
(vs_, vs, u) = solver.solution_fields()
vs_.adj_name = "vs_"
vs.adj_name = "vs"
u.adj_name = "u"
vs_.assign(ic, annotate=True)

# Store application parameters (arbitrary whether this works in
# parallel!)
directory = application_parameters["directory"]
parametersfile = File("%s/parameters.xml" % directory)
parametersfile << application_parameters

# Setup pvd storage
v_pvd = File("%s/v.pvd" % directory)
u_pvd = File("%s/u.pvd" % directory)
s_pvd = File("%s/s.pvd" % directory)

# Set-up solve
solutions = solver.solve((0, T), k_n)

# (Compute) and store solutions
begin("Solving primal")
start = time.time()
timestep_counter = 1
for (timestep, vs, u) in solutions:
    timestep_counter += 1

(v, s) = split(vs)

stop = time.time()
forward_time = (stop - start)
end()

adj_html("forward.html", "forward")
adj_html("adjoint.html", "adjoint")

#set_log_level(DEBUG)

# # Define some functional

v_obs = Function(solver.VS.sub(0).collapse())
# Read from file here!

J = Functional(inner(v - v_obs, v - v_obs)*dx*dt[FINISH_TIME])

begin("1. Computing dJdg_el")
dJdg_el = compute_gradient(J, InitialConditionParameter(g_el_var), forget=False)
file = File("%s/dJdg_el.xml.gz" % directory)
file << dJdg_el
end()

exit()

begin("2. Computing dJdg_et")
dJdg_et = compute_gradient(J, InitialConditionParameter(g_et_field),
                           forget=False)
file = File("%s/dJdg_et.xml.gz" % directory)
file << dJdg_et
end()

begin("3. Computing dJdg_il")
dJdg_il = compute_gradient(J, InitialConditionParameter(g_il_field),
                           forget=False)
file = File("%s/dJdg_il.xml.gz" % directory)
file << dJdg_il
end()

begin("4. Computing dJdg_it")
dJdg_il = compute_gradient(J, InitialConditionParameter(g_it_field),
                           forget=False)
file = File("%s/dJdg_it.xml.gz" % directory)
file << dJdg_it
end()

print "Time for forward problem: %g" % (forward_time)
print "Time for computing gradient: %g" % (gradient_time)

list_timings()
