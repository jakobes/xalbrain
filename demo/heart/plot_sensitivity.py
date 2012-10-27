from dolfin import *

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

directory = "default-adjoint-results"
e = Function(E, "%s/g_el_field_sensitivity.xml.gz" % directory)
plot(e, title="field")

e = Function(E, "%s/g_el_var_sensitivity.xml.gz" % directory)
plot(e, title= "var", interactive=True)
