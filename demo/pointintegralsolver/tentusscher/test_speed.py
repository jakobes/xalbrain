from dolfin import *
from tentusscher_2004_mcell import *
import pylab
import numpy

def main(form_compiler_parameters):

    parameters["form_compiler"].update(form_compiler_parameters)
    #info(parameters["form_compiler"], True)

    params = default_parameters(stim_start=0.0)
    state_init = init_values()

    mesh = UnitSquareMesh(5, 5)
    V = VectorFunctionSpace(mesh, "CG", 1, dim=state_init.value_size())
    u = Function(V)
    time = Constant(0.0)

    form = rhs(u, time, params)
    tstop = 10

    dt = 0.1

    Scheme = BackwardEuler

    u.interpolate(state_init)

    # Create Scheme and Solver
    scheme = Scheme(form*dP, u, time)
    solver = PointIntegralSolver(scheme)
    scheme.t().assign(0.0)

    solver.parameters.newton_solver.report = False
    solver.parameters.newton_solver.iterations_to_retabulate_jacobian = 5
    solver.parameters.newton_solver.maximum_iterations = 12

    # Take one step forward
    solver.step(dt)

    return timings(True) # True argument makes sure to reset timings

def make_bar_diagram(options, results, title):

    N = len(options)
    ind = numpy.arange(N)  # the x locations for the groups
    width = 0.35       # the width of the bars

    fig = pylab.figure(figsize=(20, 8))
    pylab.title(title)
    ax = fig.add_subplot(111)

    pylab.bar(ind, [results[op] for op in options])
    ax.set_xticks(ind+width)
    ax.set_xticklabels(options)

    pylab.ylabel("Time (s)")
    filename = "%s.pdf" % title.replace(" ", "_")
    print "Saving to %s" % filename
    pylab.savefig(filename)

if __name__ == "__main__":

    options = (#"-O0",
               #"-O2",
               "-O3",
               #"-ffast-math", # slow
               "-ffast-math -O3", # fast
               #"-march=native",
               #"-O3 -ffast-math -march=native", # fast
               #"-O3 -fno-math-errno -funsafe-math-optimizations -ffinite-math-only -fno-rounding-math -fno-signaling-nans -fcx-limited-range"
               "-O3 -fno-math-errno", # Pretty fast!
               "-O3 -ffinite-math-only",
               "-O3 -fno-signaling-nans",
               "-O3 -fcx-limited-range",
               "-O3 -funsafe-math-optimizations" # Pretty slow!
               )

    F_results = {}
    J_results = {}

    for cpp_flags in options:

        form_compiler_parameters = parameters["form_compiler"].copy()
        form_compiler_parameters["optimize"] = False
        form_compiler_parameters["cpp_optimize"] = True
        form_compiler_parameters["cpp_optimize_flags"] = cpp_flags
        form_compiler_parameters["quadrature_degree"] = 2

        times = main(form_compiler_parameters)

        tabulate_F_avg = times.get("Implicit stage: tabulate_tensor (F)",
                                   "Average time")
        tabulate_J_avg = times.get("Implicit stage: tabulate_tensor (J)",
                                   "Average time")
        tabulate_F_total = times.get("Implicit stage: tabulate_tensor (F)",
                                     "Total time")
        tabulate_J_total = times.get("Implicit stage: tabulate_tensor (J)",
                                     "Total time")

        F_results[cpp_flags] = float(tabulate_F_avg)#, float(tabulate_F_total)
        J_results[cpp_flags] = float(tabulate_J_avg)# float(tabulate_J_total)


    make_bar_diagram(options, F_results, title="tabulate_tensor (F) (avg)")
    make_bar_diagram(options, J_results, title="tabulate_tensor (J) (avg)")

    pylab.show()
