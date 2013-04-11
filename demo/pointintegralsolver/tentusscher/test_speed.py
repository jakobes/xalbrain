from dolfin import *
from tentusscher_2004_mcell import *

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

if __name__ == "__main__":

    options = ("-O0",
               "-O2",
               "-O3 -ffast-math -march=native")

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

        #info(times, True)
        print "cpp_flags = ", cpp_flags
        print "tabulate_F_avg = ", tabulate_F_avg,
        print ", tabulate_J_avg = ", tabulate_J_avg
        print "tabulate_F_total = ", tabulate_F_total,
        print ", tabulate_J_total = ", tabulate_J_total
        print
