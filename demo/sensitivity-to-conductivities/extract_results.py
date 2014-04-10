"""
Sample script to extract data from HDF5 timeseries.
"""

import os
import sys
from dolfin import *

assert len(sys.argv) > 1, "Usage: python %s results-dir" % sys.argv[0]

directory = sys.argv[1]

parameters["reorder_dofs_serial"] = False

mesh = Mesh("data/mesh115_refined.xml.gz")
num_states = 18
V = FunctionSpace(mesh, "CG", 1)
S = FunctionSpace(mesh, "CG", 1)
VS = VectorFunctionSpace(mesh, "CG", 1, dim=num_states + 1)
vs = Function(VS)
v = Function(V)
s = Function(S)

U = FunctionSpace(mesh, "CG", 1, 1)
u = Function(U)

vs_series = TimeSeriesHDF5(mesh.mpi_comm(), "%s/vs" % directory)
u_series = TimeSeriesHDF5(mesh.mpi_comm(), "%s/u" % directory)

times = vs_series.vector_times()
print "Found %d data samples in the series." % len(times)

for t in times:
    vs_series.retrieve(vs.vector(), t)
    u_series.retrieve(u.vector(), t)
    vs0 = vs.split(deepcopy=True)[0]
    vs1 = vs.split(deepcopy=True)[1]
    v.assign(vs0)
    s.assign(vs1)
    plot(v, title="v")
    plot(s, title="s0")
    plot(u, title="u")

interactive()
