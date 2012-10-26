from dolfin import *

parameters["reorder_dofs_serial"] = False # Crucial!
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True

mesh = Mesh("data/mesh115_refined.xml.gz")
mesh.coordinates()[:] /= 1000.0 # Scale mesh from micrometer to millimeter
mesh.coordinates()[:] /= 4.0    # Scale mesh as indicated by Johan

E = FunctionSpace(mesh, "CG", 1)

e = Function(E, "foo/adjoint_sensitivity_map.xml.gz")
plot(e, interactive=True)
