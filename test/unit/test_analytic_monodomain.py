import pytest


from xalbrain import (
    Model,
    NoCellModel,
    BasicSplittingSolver,
    SplittingSolver,
    BasicMonodomainSolver,
    MonodomainSolver
)

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


def main(solver, N: int, dt: float, T: float, theta: float) -> Tuple[float, float, float, float]:
    """Run monodomain MMS."""
    mesh = UnitSquareMesh(N, N)
    time = Constant(0.0)
    lam = Constant(1.0)
    cell_model = NoCellModel()

    ac_str = "(8*pi*pi*lam*sin(t) + (lam + 1)*cos(t))*cos(2*pi*x[0])*cos(2*pi*x[1])/(lam + 1)"
    stimulus = Expression(ac_str, t=time, lam=lam, degree=3)

    parameters = solver.default_parameters()
    _solver = solver(mesh, time, 1.0, I_s=stimulus, parameters=parameters)

    vs0 = Function(_solver.V)
    vs_, vs = _solver.solution_fields()
    vs_.assign(vs0)

    for timestep, (v_, v) in _solver.solve(0, T, dt):
        continue

    v_exact = Expression(
        "sin(t)*cos(2*pi*x[0])*cos(2*pi*x[1])",
        t=T,
        degree=3
    )

    # compute errors
    # v, s = vs.split(deepcopy=True)
    v_error = errornorm(v_exact, v, "L2", degree_rise=2)
    return (v_error, mesh.hmin(), dt, T)


@pytest.mark.parametrize("solver", [
    pytest.param(BasicMonodomainSolver),
    pytest.param(MonodomainSolver)
])
def test_spatial_and_temporal_convergence(solver) -> None:
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
        v_error, h, a, T = main(solver, N*(2**level), a, T, theta)
        v_errors.append(v_error)
        dts.append(a)
        hs.append(h)

    v_rates = convergence_rate(hs, v_errors)
    print("v_errors = {}".format(v_errors))
    print("v_rates = {}".format(v_rates))
    assert all(v > 1.9 for v in v_rates), "Failed convergence for v"


if __name__ == "__main__":
    test_spatial_and_temporal_convergence(MonodomainSolver)
