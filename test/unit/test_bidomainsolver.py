"""
This test solves the bidomain equations with an analytic solution to
verify the correctness of the BidomainSolver.
"""

__author__ = "Marie E. Rognes (meg@simula.no), 2013"
__all__ = []

import pytest

from cbcbeat.dolfinimport import Expression, Constant, UnitSquareMesh
from cbcbeat import BidomainSolver, errornorm
from cbcbeat.utils import convergence_rate

from testutils import medium

def main(N, dt, T, theta):

    # Create data
    mesh = UnitSquareMesh(N, N)
    time = Constant(0.0)
    ac_str = "cos(t)*cos(2*pi*x[0])*cos(2*pi*x[1]) + 4*pow(pi, 2)*cos(2*pi*x[0])*cos(2*pi*x[1])*sin(t)"
    stimulus = Expression(ac_str, t=time, degree=5)
    M_i = 1.
    M_e = 1.0

    # Set-up solver
    params = BidomainSolver.default_parameters()
    params["theta"] = theta
    solver = BidomainSolver(mesh, time, M_i, M_e, I_s=stimulus, params=params)

    # Define exact solution (Note: v is returned at end of time
    # interval(s), u is computed at somewhere in the time interval
    # depending on theta)
    v_exact = Expression("cos(2*pi*x[0])*cos(2*pi*x[1])*sin(t)", t=T, degree=3)
    u_exact = Expression("-cos(2*pi*x[0])*cos(2*pi*x[1])*sin(t)/2.0",
                         t=T - (1. - theta)*dt, degree=3)

    # Define initial condition(s)
    (v_, vu) = solver.solution_fields()

    # Solve
    solutions = solver.solve((0, T), dt)
    for (interval, fields) in solutions:
        continue

    # Compute errors
    (v, u) = vu.split(deepcopy=True)
    v_error = errornorm(v_exact, v, "L2", degree_rise=2)
    u_error = errornorm(u_exact, u, "L2", degree_rise=2)
    return [v_error, u_error, mesh.hmin(), dt, T]

@medium
def test_spatial_convergence():
    """Take a very small timestep, reduce mesh size, expect 2nd order
    convergence."""
    v_errors = []
    u_errors = []
    hs = []
    dt = 0.001
    T = 10*dt
    for N in (5, 10, 20, 40):
        (v_error, u_error, h, dt, T) = main(N, dt, T, 0.5)
        v_errors.append(v_error)
        u_errors.append(u_error)
        hs.append(h)

    v_rates = convergence_rate(hs, v_errors)
    u_rates = convergence_rate(hs, u_errors)
    print "dt, T = ", dt, T
    print "v_errors = ", v_errors
    print "u_errors = ", u_errors
    print "v_rates = ", v_rates
    print "u_rates = ", u_rates

    assert all(v > 1.99 for v in v_rates), "Failed convergence for v"
    assert all(u > 1.99 for u in u_rates), "Failed convergence for u"

@medium
def test_spatial_and_temporal_convergence():
    v_errors = []
    u_errors = []
    dts = []
    hs = []
    T = 0.5
    dt = 0.05
    theta = 0.5
    N = 5
    for level in (0, 1, 2, 3):
        a = dt/(2**level)
        (v_error, u_error, h, a, T) = main(N*(2**level), a, T, theta)
        v_errors.append(v_error)
        u_errors.append(u_error)
        dts.append(a)
        hs.append(h)

    v_rates = convergence_rate(hs, v_errors)
    u_rates = convergence_rate(hs, u_errors)
    print "v_errors = ", v_errors
    print "u_errors = ", u_errors
    print "v_rates = ", v_rates
    print "u_rates = ", u_rates

    assert v_rates[-1] > 1.95, "Failed convergence for v"
    assert u_rates[-1] > 1.9, "Failed convergence for u"
