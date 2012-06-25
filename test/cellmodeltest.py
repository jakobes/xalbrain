from dolfin import *
import pylab
import os
#from dolfin_adjoint import *

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

while (t1 <= T):
    # Solve
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

if int(os.environ.get("DOLFIN_NOPLOT", 0)) != 1:
    pylab.plot(times, v_values, 'b*')
    pylab.plot(times, s_values, 'r-')
    pylab.show()
