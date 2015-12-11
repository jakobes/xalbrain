"""
This demo demonstrates how one can use the basic cell solver for the
collection of cell models available in cbcbeat.

A single cell is considered with the initial conditions given by the
model. A stimulus of 6.25 X is applied between for t in [2, 11], and
the evolution of the variables are computed and plotted.
"""

__author__ = "Marie E. Rognes (meg@simula.no), 2013"

import math
import pylab
from cbcbeat import *
import time as systime
# For easier visualization of the variables
parameters["reorder_dofs_serial"] = False

# For computing faster
parameters["form_compiler"]["cpp_optimize"] = True
flags = "-O3 -ffast-math -march=native"
parameters["form_compiler"]["cpp_optimize_flags"] = flags


class Stimulus(Expression):
    "Some self-defined stimulus."
    def __init__(self, t):
        self.t = t
    def eval(self, value, x):
        if float(self.t) >= 0 and float(self.t) <= 1:
            value[0] = -0.2*(-100)
        else:
            value[0] = 0.0

class StimulusBR(Expression):
    "Some self-defined stimulus."
    def __init__(self, t):
        self.t = t
    def eval(self, value, x):
        if float(self.t) >= 0 and float(self.t) <= 1:
            value[0] = -0.2*(-200)
        else:
            value[0] = 0.0

def main():
    "Solve a single cell model on some time frame."

    # Initialize model and assign stimulus
    model = Beeler_reuter_1977()
    #model = Grandi_pasqualini_bers_2010()
    #model = FitzHughNagumoManual()
    #model = Fitzhughnagumo()
    #model = Tentusscher_2004_mcell()
    #model=Fenton_karma_1998_BR_altered()
    #model=Fenton_karma_1998_MLR1_altered()
    time = Constant(0.0)
    model.stimulus = StimulusBR(time)

    # Initialize solver
    """
    params = BasicSingleCellSolver.default_parameters()
    params["theta"] = 0.5
    params["nonlinear_variational_solver"]["newton_solver"]["linear_solver"] = "lu"
    solver = BasicSingleCellSolver(model, time, params)
    """
    params = SingleCellSolver.default_parameters()
    #params["theta"] = 0.5
    params["scheme"] = "GRL1"
    #params["nonlinear_variational_solver"]["newton_solver"]["linear_solver"] = "lu"
    solver = SingleCellSolver(model, time, params)
    
    
    # Assign initial conditions
    (vs_, vs) = solver.solution_fields()
    vs_.assign(model.initial_conditions())

    # Solve and extract values
    dt = 0.1
    interval = (0.0, 400.0)

    solutions = solver.solve(interval, dt)
    times = []
    values = []
    for ((t0, t1), vs) in solutions:
        #print "Current time: %g" % t1
        times.append(t1)
        values.append(vs.vector().array())

    return times, values

def plot_results(times, values, show=True):
    "Plot the evolution of each variable versus time."
    outfile=open('variables2.dat','w')

    outfile.write(str(times))
    outfile.write('\n')
    variables = zip(*values)
    pylab.figure(figsize=(20, 10))

    rows = int(math.ceil(math.sqrt(len(variables))))
    for (i, var) in enumerate(variables):
        pylab.subplot(rows, rows, i+1)
        pylab.plot(times, var, '*-')
        pylab.title("Var. %d" % i)
        pylab.xlabel("t")
        pylab.grid(True)
        outfile.write(str(var))
        outfile.write('\n')
    info_green("Saving plot to 'variables2.pdf'")
    outfile.close()
    pylab.savefig("variables2.pdf")
    if show:
        pylab.show()

if __name__ == "__main__":
    tic= systime.time()
    (times, values) = main()
    print systime.time()-tic
    plot_results(times, values, show=False)
    
    
