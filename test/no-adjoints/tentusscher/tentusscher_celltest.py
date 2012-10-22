"""
"""

# Marie E. Rognes <meg@simula.no>
# Last changed: 2012-10-22

from dolfin import *
from beatadjoint import *

from tentusscher_2004_mcell import Tentusscher_2004_mcell

class AppliedCurrent(Expression):
    def __init__(self, t=0.0):
        self.t = t
    def eval(self, value, x):
        if self.t >= 50 and self.t < 60:
            v_amp = 125
            value[0] = 0.05*v_amp
        else:
            value[0] = 0.0

cell = Tentusscher_2004_mcell()
cell.applied_current = AppliedCurrent()
solver = CellSolver(cell)

# Setup initial condition
(vs_, vs) = solver.solution_fields()
ics = project(cell.initial_conditions(), solver.VS)
vs_.assign(ics)

# Initial set-up
(T0, T) = (0, 10)
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

