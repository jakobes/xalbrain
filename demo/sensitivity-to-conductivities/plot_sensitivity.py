import sys
from dolfin import *

directory = sys.argv[1]

parameters["reorder_dofs_serial"] = False # Crucial!
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True

mesh = Mesh("data/mesh115_refined.xml.gz")
mesh.coordinates()[:] /= 1000.0 # Scale mesh from micrometer to millimeter
mesh.coordinates()[:] /= 10.0 # Scale mesh from millimeter to centimeter
mesh.coordinates()[:] /= 4.0    # Scale mesh as indicated by Johan

E = FunctionSpace(mesh, "CG", 1)

if False:
    c = Function(E)
    File("data/healthy_g_il_field.xml.gz") >> c
    plot(c, interactive=True, rescale=True)
    File("data/healthy_g_it_field.xml.gz") >> c
    plot(c, interactive=True, rescale=True)
    File("data/healthy_g_el_field.xml.gz") >> c
    plot(c, interactive=True, rescale=True)
    File("data/healthy_g_et_field.xml.gz") >> c
    plot(c, interactive=True, rescale=True)

fields = ["g_il", "g_el", "g_it", "g_et"]
for field in fields:
    e = Function(E, "%s/%s_sensitivity.xml.gz" % (directory, field))
    plot(e, title=field)
    file = File("%s/%s_sensitivity.pvd" % (directory, field))
    file << e


interactive()
