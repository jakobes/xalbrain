"""
This test just solves the bidomain equations with an analytic solution
(assuming no state variables) to verify the correctness of the
splitting solver.
"""

__author__ = "Marie E. Rognes (meg@simula.no), 2012--2013"
__all__ = []

from dolfin import *
from beatadjoint import *

if __name__ == "__main__":

    # Create domain
    level = 0
    N = 10*(2**level)
    mesh = UnitSquareMesh(N, N)

    # Create cardiac cell model
    cell_model = NoCellModel()

    # Create stimulus
    ac_str = "cos(t)*cos(2*pi*x[0])*cos(2*pi*x[1]) + 4*pow(pi, 2)*cos(2*pi*x[0])*cos(2*pi*x[1])*sin(t)"
    stimulus = Expression(ac_str, t=0, degree=5)

    # Create cardiac model
    heart = CardiacModel(mesh, 1.0, 1.0, cell_model, stimulus=stimulus)

    # Set-up solver
    parameters = SplittingSolver.default_parameters()
    parameters["enable_adjoint"] = True # FIXME
    parameters["linear_variational_solver"]["linear_solver"] = "direct"
    solver = BasicSplittingSolver(heart, parameters)
    theta = solver.parameters["theta"]

    # Define end-time and (constant) timestep
    dt = 0.01/(2**level)
    T = 0.1

    # Define exact solution (Note: v is returned at end of time
    # interval(s), u is computed at somewhere in the time interval
    # depending on theta)
    v_exact = Expression("cos(2*pi*x[0])*cos(2*pi*x[1])*sin(t)", t=T, degree=5)
    u_exact = Expression("-cos(2*pi*x[0])*cos(2*pi*x[1])*sin(t)/2.0",
                         t=T - (1 - theta)*dt, degree=5)

    # Define initial condition(s)
    vs0 = Function(solver.VS)
    (vs_, vs, u) = solver.solution_fields()
    vs_.assign(vs0)

    # Solve
    info_green("Solving primal")
    solutions = solver.solve((0, T), dt)
    for (timestep, vs, u) in solutions:
        plot(stimulus, title="stimulus", mesh=mesh)

        continue
    interactive()

    (v, s) = vs.split(deepcopy=True)

    # Pre-computed reference errors (for regression checking):
    v_reference = 4.1152719193176370e-03
    u_reference = 2.0271098018943513e-03

    # Compute errors
    v_error = errornorm(v_exact, v, "L2", degree_rise=5)
    u_error = errornorm(u_exact, u, "L2", degree_rise=5)
    v_diff = abs(v_error - v_reference)
    u_diff = abs(u_error - u_reference)
    tolerance = 1.e-10
    msg = "Maximal %s value does not match reference: diff is %.16e"
    print "v_error = %.16e" % v_error
    print "u_error = %.16e" % u_error
    assert (v_diff < tolerance), msg % ("v", v_diff)
    assert (u_diff < tolerance), msg % ("u", u_diff)
