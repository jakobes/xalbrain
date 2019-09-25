"""
This test solves the bidomain equations (assuming no state variables)
with an analytic solution to verify the correctness of the basic
splitting solver.
"""

__author__ = "Marie E. Rognes (meg@simula.no), 2012--2013"
__all__ = []


import pytest

from xalbrain import (
    BrainModel,
    BidomainSplittingSolver,
    SplittingSolverParameters,
    BidomainParameters,
    ODESolverParameters,
    BidomainSolver,
    ODESolver,
)

from xalbrain.cellmodels import NoCellModel

from dolfin import (
    Constant,
    UnitSquareMesh,
    Function,
    Expression,
    errornorm,
)

from dolfin import __version__ as dolfin_version

from xalbrain.utils import convergence_rate

from testutils import slow
from typing import Tuple


def main(*, N: int, dt: float, T: float, theta: float) -> Tuple[float, float, float, float, float]:
    # Create cardiac model
    mesh = UnitSquareMesh(N, N)
    time = Constant(0.0)
    cell_model = NoCellModel()

    ac_str = "cos(t)*cos(2*pi*x[0])*cos(2*pi*x[1]) + 4*pow(pi, 2)*cos(2*pi*x[0])*cos(2*pi*x[1])*sin(t)"
    stimulus = Expression(ac_str, t=time, degree=5)

    brain = BrainModel(
        time=time,
        mesh=mesh,
        cell_model=cell_model,
        intracellular_conductivity=1,
        other_conductivity=1,
        external_stimulus=stimulus
    )

    # Set-up solver
    splitting_parameters = SplittingSolverParameters(theta=theta)
    bidomain_parameters = BidomainParameters(linear_solver_type="direct")
    ode_parameters = ODESolver.default_parameters()

    solver = BidomainSplittingSolver(
        brain=brain,
        parameters=splitting_parameters,
        ode_parameters=ode_parameters,
        pde_parameters=bidomain_parameters
    )

    # Define exact solution (Note: v is returned at end of time
    # interval(s), u is computed at somewhere in the time interval
    # depending on theta)

    v_exact = Expression("cos(2*pi*x[0])*cos(2*pi*x[1])*sin(t)", t=T, degree=5)
    u_exact = Expression(
        "-cos(2*pi*x[0])*cos(2*pi*x[1])*sin(t)/2.0",
        t=T - (1 - theta)*dt,
        degree=5
    )

    # Define initial condition(s)
    vs0 = Function(solver.VS)
    vs_, *_ = solver.solution_fields()
    vs_.assign(vs0)

    # Solve
    for solution_struct in solver.solve(0, T, dt):
        continue

    # Compute errors
    v, s = solution_struct.vs.split(deepcopy=True)
    v, u, r = solution_struct.vur.split(deepcopy=True)
    v_error = errornorm(v_exact, v, "L2", degree_rise=5)
    u_error = errornorm(u_exact, u, "L2", degree_rise=5)
    return v_error, u_error, mesh.hmin(), dt, T


@slow
@pytest.mark.xfail(dolfin_version == "2016.2.0", reason="Unknown")
def test_spatial_and_temporal_convergence() -> None:
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
        v_error, u_error, h, a, T = main(N=N*(2**level), dt=a, T=T, theta=theta)
        v_errors.append(v_error)
        u_errors.append(u_error)
        dts.append(a)
        hs.append(h)

    v_rates = convergence_rate(hs=hs, errors=v_errors)
    u_rates = convergence_rate(hs=hs, errors=u_errors)
    print("v_errors = ", v_errors)
    print("u_errors = ", u_errors)
    print("v_rates = ", v_rates)
    print("u_rates = ", u_rates)

    assert all(v > 1.9 for v in v_rates), "Failed convergence for v"
    assert all(u > 1.9 for u in u_rates), "Failed convergence for u"


if __name__ == "__main__":
    test_spatial_and_temporal_convergence()
