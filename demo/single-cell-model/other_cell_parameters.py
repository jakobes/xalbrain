"""
"""

__author__ = "Marie E. Rognes (meg@simula.no), 2017"

import math
import pylab
from cbcbeat import *
from demo import plot_results

parameters["adjoint"]["stop_annotating"] = True

# For easier visualization of the variables
parameters["reorder_dofs_serial"] = False

# For computing faster
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True
flags = "-O3 -ffast-math -march=native"
parameters["form_compiler"]["cpp_optimize_flags"] = flags

class Stimulus(Expression):
    "Some self-defined stimulus."
    def __init__(self, **kwargs):
        self.t = kwargs["time"]
    def eval(self, value, x):
        if float(self.t) >= 2 and float(self.t) <= 11:
            v_amp = 125
            value[0] = 0.05*v_amp
        else:
            value[0] = 0.0

def main(scenario="default"):
    "Solve a single cell model on some time frame."

    # Initialize model and assign stimulus
    params = Tentusscher_panfilov_2006_epi_cell.default_parameters()
    if scenario is not "default":
        new = {"g_Na": params["g_Na"]*(1-0.62),
               "g_CaL": params["g_CaL"]*(1-0.69),
               "g_Kr": params["g_Kr"]*(1-0.70),
               "g_K1": params["g_K1"]*(1-0.80)}
        model = Tentusscher_panfilov_2006_epi_cell(params=new)
    else:
        model = Tentusscher_panfilov_2006_epi_cell()
        
    time = Constant(0.0)
    model.stimulus = Stimulus(time=time, degree=0)

    # Initialize solver
    params = SingleCellSolver.default_parameters()
    params["scheme"] = "GRL1"
    solver = SingleCellSolver(model, time, params)

    # Assign initial conditions
    (vs_, vs) = solver.solution_fields()
    vs_.assign(model.initial_conditions())

    # Solve and extract values
    dt = 0.1
    interval = (0.0, 800.0)

    solutions = solver.solve(interval, dt)
    times = []
    values = []
    for ((t0, t1), vs) in solutions:
        print "Current time: %g" % t1
        times.append(t1)
        values.append(vs.vector().array())

    return times, values

def compare_results(times, many_values, legends=(), show=True):
    "Plot the evolution of each variable versus time."

    pylab.figure(figsize=(20, 10))
    for values in many_values:
        variables = zip(*values)
        rows = int(math.ceil(math.sqrt(len(variables))))
        for (i, var) in enumerate([variables[0],]):
            #pylab.subplot(rows, rows, i+1)
            pylab.plot(times, var, '-')
            pylab.title("Var. %d" % i)
            pylab.xlabel("t")
            pylab.grid(True)

    pylab.legend(legends)
    info_green("Saving plot to 'variables.pdf'")
    pylab.savefig("variables.pdf")
    if show:
        pylab.show()

if __name__ == "__main__":

    (times, values1) = main("default")
    (times, values2) = main("gray zone")
    compare_results(times, [values1, values2], legends=("default", "gray zone"),
                    show=True)
