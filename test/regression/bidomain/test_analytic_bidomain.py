"""
This test just solves the bidomain equations with an analytic solution
(assuming no state variables) to verify the correctness of the
splitting solver.
"""

__author__ = "Marie E. Rognes (meg@simula.no), 2012--2013"
__all__ = []

from dolfin import *
from beatadjoint import *

level = 0

class MyHeart(CardiacModel):
    def __init__(self, cell_model):
        CardiacModel.__init__(self, cell_model)
    def domain(self):
        N = 10*(2**level)
        return UnitSquareMesh(N, N)
    def conductivities(self):
        M_i = 1.0
        M_e = 1.0
        return (M_i, M_e)

if __name__ == "__main__":

    # Set-up model
    cell = NoCellModel()
    heart = MyHeart(cell)
    ac_str = "cos(t)*cos(2*pi*x[0])*cos(2*pi*x[1]) + 4*pow(pi, 2)*cos(2*pi*x[0])*cos(2*pi*x[1])*sin(t)"
    heart.stimulus = Expression(ac_str, t=0, degree=5)

    # Set-up solver
    parameters = SplittingSolver.default_parameters()
    parameters["enable_adjoint"] = True # FIXME
    parameters["linear_variational_solver"]["linear_solver"] = "direct"
    solver = SplittingSolver(heart, parameters)
    theta = solver.parameters["theta"]

    # Define end-time and (constant) timestep
    T = 0.1
    dt = 0.01/(2**level)

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
        continue

    (v, s) = vs.split()

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