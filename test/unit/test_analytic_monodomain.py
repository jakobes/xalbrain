import pytest


from xalbrain import (
    BrainModel,
    MonodomainSplittingSolver,
    MonodomainParameters,
    SplittingSolverParameters,
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

from xalbrain.utils import (
    convergence_rate,
)

from typing import (
    Tuple,
)


def main(*, N: int, dt: float, T: float, theta: float) -> Tuple[float, float, float, float]:
    """Run monodomain MMS."""
    mesh = UnitSquareMesh(N, N)
    time = Constant(0.0)
    lam = Constant(1.0)
    cell_model = NoCellModel()

    ac_str = "(8*pi*pi*lam*sin(t) + (lam + 1)*cos(t))*cos(2*pi*x[0])*cos(2*pi*x[1])/(lam + 1)"
    stimulus = Expression(ac_str, t=time, lam=lam, degree=3)
    brain = BrainModel(
        time=time,
        mesh=mesh,
        cell_model=cell_model,
        intracellular_conductivity=1,
        other_conductivity=lam,
        external_stimulus=stimulus
    )

    # Define solver solver
    splitting_parameters = SplittingSolverParameters(theta=theta)
    bidomain_parameters = MonodomainParameters(linear_solver_type="direct")
    ode_parameters = ODESolver.default_parameters()

    solver = MonodomainSplittingSolver(
        brain=brain,
        parameters=splitting_parameters,
        ode_parameters=ode_parameters,
        pde_parameters=bidomain_parameters
    )

    vs0 = Function(solver.VS)
    vs_, vs, vur = solver.solution_fields()
    vs_.assign(vs0)

    for solution_struct in solver.solve(0, T, dt):
        continue

    v_exact = Expression(
        "sin(t)*cos(2*pi*x[0])*cos(2*pi*x[1])",
        t=T,
        degree=3
    )

    # compute errors
    v, s = vs.split(deepcopy=True)
    v_error = errornorm(v_exact, v, "L2", degree_rise=2)
    return (v_error, mesh.hmin(), dt, T)


def test_analytic_mondomain() -> None:
    """Test errors vs reference"""
    N = 20
    dt = 1e-6
    T = 10*dt

    v_error, h, dt, T = main(N, dt, T, 0.5)
    v_reference = 4.203505851866419e-08
    v_diff = abs(v_error - v_reference)
    tol = 1e-9

    print("v_diff: {}".format(v_diff))
    assert v_diff < tol, "Failed error for u, diff: {}".format(v_diff)


def test_spatial_and_temporal_convergence() -> None:
    """Test convergence rates for bidomain solver."""
    v_errors = []
    dts = []
    hs = []

    dt = 1e-3
    theta = 0.5
    N = 5

    for level in range(3):
        a = dt/(2**level)
        T = 10*a
        v_error, h, a, T = main(N=N*(2**level), dt=a, T=T, theta=theta)
        v_errors.append(v_error)
        dts.append(a)
        hs.append(h)

    v_rates = convergence_rate(hs=hs, errors=v_errors)
    print("v_errors = {}".format(v_errors))
    print("v_rates = {}".format(v_rates))
    assert all(v > 1.9 for v in v_rates), "Failed convergence for v"


if __name__ == "__main__":
    args = {
        "N": 10,
        "dt": 5e-3,
        "T": 1,
        "theta": 1.0,
    }

    # main(**args)
    # test_analytic_mondomain()
    test_spatial_and_temporal_convergence()
