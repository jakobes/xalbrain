"""
This test solves the bidomain equations (assuming no state variables)
with an analytic solution to verify the correctness of the basic
splitting solver.
"""

__author__ = "Marie E. Rognes (meg@simula.no), 2012--2013"
__all__ = []

import pytest

from xalbrain import (
    Model,
    NoCellModel,
    BasicSplittingSolver,
    SplittingSolver,
    BasicBidomainSolver,
    BidomainSolver,
)

from dolfin import (
    Constant,
    UnitSquareMesh,
    Function,
    Expression,
    errornorm,
)

from dolfin import __version__ as dolfin_version

from xalbrain.utils import convergence_rate

from typing import Tuple


import dolfin
dolfin.set_log_level(100)


def main(
    solver,
    N: int,
    dt: float,
    T: float,
    theta: float
) -> Tuple[float, float, float, float, float]:
    # Create cardiac model
    mesh = UnitSquareMesh(N, N)
    time = Constant(0.0)
    cell_model = NoCellModel()

    ac_str = "cos(t)*cos(2*pi*x[0])*cos(2*pi*x[1]) + 4*pow(pi, 2)*cos(2*pi*x[0])*cos(2*pi*x[1])*sin(t)"
    stimulus = Expression(ac_str, t=time, degree=5)

    ps = solver.default_parameters()
    _solver = solver(mesh, time, 1.0, 1.0, stimulus, parameters=ps)

    # Define exact solution (Note: v is returned at end of time
    # interval(s), u is computed at somewhere in the time interval
    # depending on theta)

    v_exact =  Expression("cos(2*pi*x[0])*cos(2*pi*x[1])*sin(t)", t=T, degree=5)
    u_exact = Expression(
        "-cos(2*pi*x[0])*cos(2*pi*x[1])*sin(t)/2.0",
        t=T - (1 - theta)*dt,
        degree=5
    )

    # Define initial condition(s)
    vs0 = Function(_solver.V)
    vs_, *_ = _solver.solution_fields()
    vs_.assign(vs0)

    # Solve
    for _, (vs_, vur) in _solver.solve(0, T, dt):
        continue

    # Compute errors
    v, u, *_ = vur.split(deepcopy=True)
    v_error = errornorm(v_exact, v, "L2", degree_rise=5)
    u_error = errornorm(u_exact, u, "L2", degree_rise=5)
    return v_error, u_error, mesh.hmin(), dt, T


@pytest.mark.xfail(dolfin_version == "2016.2.0", reason="Unknown")
@pytest.mark.parametrize("solver", [
    pytest.param(BasicBidomainSolver),
    pytest.param(BidomainSolver)
])
def test_spatial_and_temporal_convergence(solver) -> None:
    """Test convergence rates for bidomain solver."""
    v_errors = []
    u_errors = []
    dts = []
    hs = []
    T = 0.1
    dt = 0.01
    theta = 0.5
    N = 10
    for level in (1, 2, 3):
        a = dt/(2**level)
        v_error, u_error, h, a, T = main(solver, N*(2**level), a, T, theta)
        v_errors.append(v_error)
        u_errors.append(u_error)
        dts.append(a)
        hs.append(h)

    v_rates = convergence_rate(hs, v_errors)
    u_rates = convergence_rate(hs, u_errors)
    print("v_errors = ", v_errors)
    print("u_errors = ", u_errors)
    print("v_rates = ", v_rates)
    print("u_rates = ", u_rates)

    assert all(v > 1.9 for v in v_rates), "Failed convergence for v"
    assert all(u > 1.9 for u in u_rates), "Failed convergence for u"


if __name__ == "__main__":
    test_spatial_and_temporal_convergence(BasicBidomainSolver)
