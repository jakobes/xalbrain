"""This test solves the monodomain equations with an analytic solution to verify the 
correctness of the solver."""

import pytest

from dolfin import (
    Expression,
    Constant,
    UnitSquareMesh,
    parameters,
    errornorm,
    set_log_level
)

from xalbrain import (
    MonodomainSolver,
    BasicMonodomainSolver,
)

from xalbrain.utils import convergence_rate

from typing import Tuple

import sys
from IPython import embed


set_log_level(100)


def main(N: int, dt: float, T: float, theta: float) -> Tuple[float, float, float, float]:
    """Set up solver and return errors and mesh size."""
    mesh = UnitSquareMesh(N, N)
    time = Constant(0.0)
    ac_str = "(8*pi*pi*lam*sin(t) + (lam + 1)*cos(t))*cos(2*pi*x[0])*cos(2*pi*x[1])/(lam + 1)"
    stimulus = Expression(ac_str, t=time, lam=Constant(1), degree=3)
    M_i = Constant(1.0)
    # Set up solver
    params = BasicMonodomainSolver.default_parameters()
    params["theta"] = theta
    params["linear_variational_solver"]["linear_solver"] = "direct"
    params["enable_adjoint"] = False
    solver = BasicMonodomainSolver(mesh, time, M_i, I_s=stimulus, params=params)

    v_exact  = Expression("sin(t)*cos(2*pi*x[0])*cos(2*pi*x[1])", t=T, degree=3)

    # Define initial conditions
    v_, v = solver.solution_fields()

    # Solve
    solutions = solver.solve((0, T), dt)
    for interval, fields in solutions:
        continue

    # Compute errors
    v_error = errornorm(v_exact, v, "L2", degree_rise=2)
    return v_error, mesh.hmin(), dt, T


def test_spatial_convergence() -> None:
    """Take a very small time step, reduce mesh size, expect 2nd order convergence."""
    v_errors = []
    hs = []
    dt = 1e-6
    T = 10*dt

    for N in (5, 10, 20, 40):
        v_error, h, *_ = main(N, dt, T, theta=0.5)
        v_errors.append(v_error)
        hs.append(h)

    v_rates = convergence_rate(hs, v_errors)
    print("dt, T = {dt}, {T}".format(dt=dt, T=T))
    print("v_errors = ", v_errors)
    print("v_rates = ", v_rates)

    msg = "Failed convergence for v. v_rates = {}".format(", ".join(map(str, v_rates)))
    assert all(v > 2 for v in v_rates), msg



def test_spatial_and_temporal_convergence() -> None:
    v_errors = []
    hs = []
    dt = 1e-3
    N = 5

    for level in range(3):
        a = dt/2**level
        T= 10*a
        v_error, h, *_ = main(N*2**level, a, T, theta=0.5)
        v_errors.append(v_error)
        hs.append(h)

    v_rates = convergence_rate(hs, v_errors)
    print(v_errors)
    print(v_rates)
    msg = "Failed convergence for v, v_rates: {}".format(", ".join(map(str, v_rates)))
    assert v_rates[-1] > 2, msg


if __name__ == "__main__":
    test_spatial_convergence()
    test_spatial_and_temporal_convergence()
