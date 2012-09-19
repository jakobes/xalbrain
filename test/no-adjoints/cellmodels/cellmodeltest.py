"""
Regression and correctness test for FitzHughNagumo model and pure
CellSolver: compare (in eyenorm) time evolution with results from
Section 2.4.1 p. 36 in Sundnes et al, 2006 (checked 2012-09-19), and
check that maximal v/s values do not regress
"""

# Marie E. Rognes <meg@simula.no>
# Last changed: 2012-09-19

from dolfin import *

# Cardiac solver specific imports
from beatadjoint import *
from beatadjoint.models import *

class AppliedCurrent(Expression):
    def __init__(self, t=0.0):
        self.t = t
    def eval(self, value, x):
        if self.t >= 50 and self.t < 60:
            v_amp = 125
            value[0] = 0.05*v_amp
        else:
            value[0] = 0.0

cell = FitzHughNagumo()
cell.applied_current = AppliedCurrent()
solver = CellSolver(cell)

# Setup initial condition
(vs_, vs) = solver.solution_fields()
vs_.vector()[0] = -85. # Initial condition resting state
vs_.vector()[1] = 0.

# Initial set-up
(T0, T) = (0, 400)
dt = 1.0
t0 = T0; t1 = T0 + dt

times = []
v_values = []
s_values = []

# Solve
while (t1 <= T):
    info_blue("Solving on t = (%g, %g)" % (t0, t1))
    timestep = (t0, t1)
    times += [(t0 + t1)/2]
    tmp = solver.step(timestep, vs_)
    vs.assign(tmp)

    v_values += [vs.vector()[0]]
    s_values += [vs.vector()[1]]

    # Update
    vs_.assign(vs)
    t0 = t1; t1 = t0 + dt

# Regression test
v_max_reference = 2.3839115023509514e+01
s_max_reference = 6.9925836316850834e+01
tolerance = 1.e-14
msg = "Maximal %s value does not match reference: diff is %.16e"
v_diff = abs(max(v_values) - v_max_reference)
s_diff = abs(max(s_values) - s_max_reference)
assert (v_diff < tolerance), msg % ("v", v_diff)
assert (s_diff < tolerance), msg % ("s", s_diff)

# Correctness test
import os
if int(os.environ.get("DOLFIN_NOPLOT", 0)) != 1:
    import pylab
    pylab.plot(times, v_values, 'b*')
    pylab.plot(times, s_values, 'r-')
    pylab.show()
