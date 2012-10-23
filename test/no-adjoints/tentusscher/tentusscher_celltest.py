"""
Test running-ness of tenTusscher cell model
"""

# Marie E. Rognes <meg@simula.no>
# Last changed: 2012-10-23

from dolfin import *
from beatadjoint import *

parameters["reorder_dofs"] = False
parameters["form_compiler"]["cpp_optimize"] = True

class Stimulus(Expression):
    def __init__(self, t=0.0):
        self.t = t
    def eval(self, value, x):
        if self.t >= 1 and self.t < 2:
            v_amp = 125
            value[0] = 0.05*v_amp
        else:
            value[0] = 0.0

cell = Tentusscher_2004_mcell()
cell.stimulus = Stimulus()
solver = CellSolver(cell)

# Setup initial condition
(vs_, vs) = solver.solution_fields()
ics = project(cell.initial_conditions(), solver.VS)
vs_.assign(ics)

# # Initial set-up
(T0, T) = (0, 2)
dt = 0.1
t0 = T0; t1 = T0 + dt

# Solve
times = []
v_values = []
while (t1 <= T):
    info_blue("Solving on t = (%g, %g)" % (t0, t1))
    timestep = (t0, t1)
    times += [(t0 + t1)/2]
    tmp = solver.step(timestep, vs_)
    vs.assign(tmp)
    v_values += [vs.vector()[0]]

    # Update
    vs_.assign(vs)
    t0 = t1; t1 = t0 + dt

# Print v values
print v_values

# Plot values
#import pylab
#pylab.figure()
#pylab.plot(times, v_values)
#pylab.show()

