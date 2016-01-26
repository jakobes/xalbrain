from dolfin import *
from tentusscher_2004_mcell import *
import pylab
import numpy

def main(form_compiler_parameters):

    parameters["form_compiler"].update(form_compiler_parameters)
    #info(parameters["form_compiler"], True)

    params = default_parameters(stim_start=0.0)
    state_init = init_values()

    mesh = UnitSquareMesh(10, 10)
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

def make_bar_diagram(options, results, title, normalize=False):

    N = len(options)
    ind = numpy.arange(N)  # the x locations for the groups
    width = 0.35           # the width of the bars

    fig = pylab.figure(figsize=(20, 8))
    pylab.title(title)
    ax = fig.add_subplot(111)

    if normalize:
        maximal = max(results.values())
        pylab.ylabel("Normalized time")
    else:
        maximal = 1.0
        pylab.ylabel("Time (s)")

    pylab.bar(ind, [results[op]/maximal for op in options])
    ax.set_xticks(ind+width)
    ax.set_xticklabels(options)
    for item in ax.get_xticklabels():
        item.set_fontsize(10)

    filename = "%s.pdf" % title.replace(" ", "_")
    print "Saving to %s" % filename
    pylab.savefig(filename)

def pretty_print_results(results, normalize=False):
    sorted_by_value = sorted(results, key=results.get)
    if normalize:
        maximal = max(results.values())
    else:
        maximal = 1.0
    print "Timings:\t (Flag)"
    for flag in sorted_by_value:
        print "%1.4f\t\t (%s)" % (results[flag]/maximal, flag)

if __name__ == "__main__":

    options = (#"-O0", # Slow
               "-O2", # A little less slow
               "-O3", # A little less fast than O2
               "-ffast-math", # slow
               #"-ffast-math -O3", # fast
               #"-march=native", # Slower than O0
               "-O3 -ffast-math -march=native", # Fastest
               #"-O3 -ffast-math -march=native -mtune=native", # As above
               #"-O2 -ffast-math -march=native", # Essentially just as fast
               #"-O3 -fno-math-errno", # Comparably fast (x1.3 - 3)
               "-O3 -fno-math-errno -march=native", # Comparably fast (x1.3 - 3)
               #"-O3 -ffinite-math-only", # About as slow as -O2
               #"-O3 -fno-signaling-nans", # A bit worse than the above
               #"-O3 -fcx-limited-range", # As slow as the above
               "-O3 -funsafe-math-optimizations" # Faster, but pretty slow
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

        F_results[cpp_flags] = float(tabulate_F_avg)# float(tabulate_F_total)
        J_results[cpp_flags] = float(tabulate_J_avg)# float(tabulate_J_total)

    print "Results for F (Normalized timings)"
    pretty_print_results(F_results, True)
    print

    print "Results for J (Normalized timings)"
    pretty_print_results(J_results, True)
    print

    make_bar_diagram(options, F_results, title="tabulate_tensor (F) (avg)")
    make_bar_diagram(options, J_results, title="tabulate_tensor (J) (avg)")

    #pylab.show()
