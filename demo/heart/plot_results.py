import sys
from dolfin import *

parameters["reorder_dofs_serial"] = False # Crucial!
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True

mesh = Mesh("data/mesh115_refined.xml.gz")
mesh.coordinates()[:] /= 1000.0 # Scale mesh from micrometer to millimeter
mesh.coordinates()[:] /= 10.0 # Scale mesh from millimeter to centimeter
mesh.coordinates()[:] /= 4.0    # Scale mesh as indicated by Johan

V = FunctionSpace(mesh, "CG", 1)
S = FunctionSpace(mesh, "DG", 0)
W = V*S

directory = sys.argv[1]

u_plot = Function(V)
v_plot = Function(W.sub(0).collapse())
s_plot = Function(W.sub(1).collapse())
for i in range(2,3):
    print "i = ", i
    vs = Function(W, "%s/vs_%d.xml.gz" % (directory, i))
    u = Function(V, "%s/u_%d.xml.gz" % (directory, i))
    (v, s) = vs.split(deepcopy=True)
    u_plot.assign(u)
    v_plot.assign(v)
    s_plot.assign(s)

    plot(u_plot, title="u")
    plot(v_plot, title="v")
    plot(s_plot, title="s")

interactive()
