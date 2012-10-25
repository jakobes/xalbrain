"""
"""

# Marie E. Rognes <meg@simula.no>
# Last changed: 2012-10-25

import math
from dolfin import *
from dolfin_adjoint import *
from beatadjoint import *

import time

set_log_level(PROGRESS)

# Setup application parameters and parse from command-line
application_parameters = Parameters("Application")
application_parameters.add("T", 0.5)         # End time  (ms)
application_parameters.add("timestep", 0.05) # Time step (ms)
application_parameters.add("directory", "default-results")
application_parameters.add("backend", "PETSc")
application_parameters.add("store_primal", False)
application_parameters.parse()
info(application_parameters, True)

# Update backend from application parameters
parameters["linear_algebra_backend"] = application_parameters["backend"]

# Adjust some general parameters
parameters["reorder_dofs"] = False # Crucial!
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True

# Generic cardiac parameters
#chi = 2000.0   # Membrane surface-to-volume ratio (1/cm), value from book
# NB : If not 1 => must scale ionic current! see book p. 55
#C_m = 1.0      # Membrane capacitance per unit area (micro F/(cm^2))

# Domain
mesh = Mesh("data/mesh115_refined.xml.gz")
mesh.coordinates()[:] /= 1000.0 # Scale mesh from micrometer to millimeter
mesh.coordinates()[:] /= 4.0    # Scale mesh as indicated by Johan

# Time and time-step
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

# Extract conducitivity data
V = FunctionSpace(mesh, "CG", 1)
g_el_field = Function(V, "data/g_el_field.xml.gz")
g_et_field = Function(V, "data/g_et_field.xml.gz")
g_il_field = Function(V, "data/g_il_field.xml.gz")
g_it_field = Function(V, "data/g_it_field.xml.gz")

# Construct conductivity tensors from directions and conductivity
# values relative to that coordinate system
A = as_matrix([[fiber[0], sheet[0], cross_sheet[0]],
               [fiber[1], sheet[1], cross_sheet[1]],
               [fiber[2], sheet[2], cross_sheet[2]]])
M_e_star = diag(as_vector([g_el_field, g_et_field, g_et_field]))
M_i_star = diag(as_vector([g_il_field, g_it_field, g_it_field]))
M_e = A*M_e_star*A.T
M_i = A*M_i_star*A.T

class MyHeart(CardiacModel):
    def __init__(self, cell_model):
        CardiacModel.__init__(self, cell_model)
    def domain(self):
        return mesh
    def conductivities(self):
        return (M_i, M_e)

# Setup cell and cardiac model
cell = OriginalFitzHughNagumo()
heart = MyHeart(cell)

# Define some simulation protocol (use cpp expression for speed)
stimulation_cells = MeshFunction("uint", mesh, "data/stimulation_cells.xml.gz")
from stimulation import cpp_stimulus
pulse = Expression(cpp_stimulus)
pulse.cell_data = stimulation_cells
pulse.amplitude = 7.0 # Ampere
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

# Solve
begin("Solving primal")
start = time.time()
solutions = solver.solve((0, T), k_n)

timestep_counter = 1
directory = application_parameters["directory"]

# Store application parameters
parametersfile = File("%s/parameters.xml" % directory)
parametersfile << application_parameters

# (Compute) and store solutions
store_primal = application_parameters["store_primal"]
for (timestep, vs, u) in solutions:
    if store_primal:
        vsfile = File("%s/vs_%d.xml.gz" % (directory, timestep_counter))
        vsfile << vs
        ufile = File("%s/u_%d.xml.gz" % (directory, timestep_counter))
        ufile << u
        timestep_counter += 1
    else:
        continue

stop = time.time()
forward_time = (stop - start)
end()

(v, s) = split(vs)

# # Define some functional
J = Functional(inner(v, v)*ds*dt)
begin("Computing gradient")
start = time.time()
dJdm = compute_gradient(J, InitialConditionParameter(g_el_field))
stop = time.time()
gradient_time = (stop - start)
end()

print "Time for forward problem: %g" % (forward_time)
print "Time for computing gradient: %g" % (gradient_time)

begin("Storing gradient")
sensfile = File("%s/adjoint_sensitivity_map.xml.gz" % directory)
sensfile << dJdm
end()

