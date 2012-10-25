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
application_parameters.add("directory", "default-results") # Time step (ms)
application_parameters.parse()
info(application_parameters, True)

parameters["reorder_dofs"] = False # Crucial!
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True

# Generic cardiac parameters
chi = 2000.0   # Membrane surface-to-volume ratio (1/cm), value from book
# NB : If not 1 => must scale ionic current! see book p. 55
C_m = 1.0      # Membrane capacitance per unit area (micro F/(cm^2))

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

# Load ischemic region (scalar function between 0-1, where 0 is ischemic)
V = FunctionSpace(mesh, "CG", 1)
ischemic = Function(V)
File("data/ischemic_region.xml.gz") >> ischemic
ischemic_array = ischemic.vector().array()

# Healthy and ischemic conductivities
# (All values in mS/cm (milli-Siemens per centimeter)

# Extracellular:
g_el = 6.25/(C_m*chi) # Fiber
g_et = 2.36/(C_m*chi) # Sheet
g_et = 2.36/(C_m*chi) # Cross-sheet

# Intracellular:
g_il = 1.74/(C_m*chi)   # Fiber
g_it = 0.192/(C_m*chi)  # Sheet
g_it = 0.192/(C_m*chi)  # Cross-sheet

# Extracellular:
g_el_isch = 3.125/(C_m*chi) # Fiber
g_et_isch = 1.18/(C_m*chi) # Sheet
g_et_isch = 1.18/(C_m*chi) # Cross-sheet

# Intracellular:
g_il_isch = 0.125/(C_m*chi)  # Fiber
g_it_isch = 0.125/(C_m*chi)  # Sheet
g_it_isch = 0.125/(C_m*chi)  # Cross-sheet

# Combine info into 2x2 distinct conductivity functions:
g_el_field = Function(V)
g_el_field.vector()[:] = (1-ischemic_array)*g_el_isch+ischemic_array*g_el
g_et_field = Function(V)
g_et_field.vector()[:] = (1-ischemic_array)*g_et_isch+ischemic_array*g_et
g_il_field = Function(V)
g_il_field.vector()[:] = (1-ischemic_array)*g_il_isch+ischemic_array*g_il
g_it_field = Function(V)
g_it_field.vector()[:] = (1-ischemic_array)*g_it_isch+ischemic_array*g_it

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
cpp_stimulus = """
class Stimulus : public Expression
{
public:

  boost::shared_ptr<MeshFunction<unsigned int> > cell_data;

  Stimulus() : Expression(), amplitude(0), duration(0), t(0)
  {
  }

  void eval(Array<double>& values, const Array<double>& x,
            const ufc::cell& c) const
  {
    assert(cell_data);
    const Cell cell(cell_data->mesh(), c.index);
    switch ((*cell_data)[cell.index()])
    {
    case 0:
      values[0] = 0.0;
      break;
    case 1:
      if (t <= duration)
        values[0] = amplitude;
      else
        values[0] = 0.0;
      break;
    default:
      values[0] = 0.0;
    }
  }
  double amplitude;
  double duration;
  double t;
};"""

# Define pulse stimulus
pulse = Expression(cpp_stimulus)
pulse.cell_data = stimulation_cells
pulse.amplitude = 7.0 # Ampere
pulse.duration = 10.0 # ms
pulse.t = 0.0         # ms

# Assign pulse as stimulus to heart
heart.stimulus = pulse

# Set-up solver
Solver = SplittingSolver
ps = Solver.default_parameters()
ps["enable_adjoint"] = True
ps["linear_variational_solver"]["linear_solver"] = "direct"
solver = Solver(heart, parameters=ps)

# Define initial condition here (no need to annotate this step)
ic = project(cell.initial_conditions(), solver.VS, annotate=False)

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
for (timestep, vs, u) in solutions:
    vsfile = File("%s/vs_%d.xml.gz" % (directory, timestep_counter))
    vsfile << vs
    ufile = File("%s/u_%d.xml.gz" % (directory, timestep_counter))
    ufile << u
    timestep_counter += 1

stop = time.time()
print "Time elapsed: %g" % (stop - start)
end()

list_timings()
