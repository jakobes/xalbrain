from beatadjoint import *
import os

mesh = Mesh("data/mesh115_refined.xml.gz")
mesh.coordinates()[:] /= 1000.0 # Scale mesh from micrometer to millimeter
mesh.coordinates()[:] /= 10.0   # Scale mesh from millimeter to centimeter
mesh.coordinates()[:] /= 4.0    # Scale mesh as indicated by Johan/Molly

# Set-up fundtion spaces

versions = [sys.argv[1], sys.argv[2]]

values_0 = []
values_1 = []

s_values_0 = []
s_values_1 = []

cs = range(int(sys.argv[3]))
for c in cs:
    V = FunctionSpace(mesh, "CG", 1)
    S = V
    VS = V*S

    print "Computing averages for timestep %d" % c
    filename = os.path.join(versions[0], "vs_%d.xml.gz" % c)
    vs = Function(VS, filename)
    avg_v_0 = assemble(vs[0]*vs[0]*dx)
    values_0.append(avg_v_0)
    avg_s_0 = assemble(vs[1]*vs[1]*dx)
    s_values_0.append(avg_s_0)

    S = FunctionSpace(mesh, "DG", 0)
    VS = V*S
    filename = os.path.join(versions[1], "vs_%d.xml.gz" % c)
    vs = Function(VS, filename)
    avg_v_1 = assemble(vs[0]*vs[0]*dx)
    values_1.append(avg_v_1)
    avg_s_1 = assemble(vs[1]*vs[1]*dx)
    s_values_1.append(avg_s_1)

import pylab
pylab.figure()
pylab.plot(cs, values_0, 'r*-')
pylab.plot(cs, values_1, 'b*-')
pylab.xlabel("c")
pylab.ylabel("v*v*dx")
pylab.grid(True)
pylab.legend(versions)

pylab.figure()
pylab.plot(cs, s_values_0, 'r*-')
pylab.plot(cs, s_values_1, 'b*-')
pylab.xlabel("c")
pylab.ylabel("s*s*dx")
pylab.grid(True)
pylab.legend(versions)

pylab.show()
