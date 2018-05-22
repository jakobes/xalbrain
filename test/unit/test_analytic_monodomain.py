import pytest


from xalbrain import (
    CardiacModel,
    NoCellModel,
    BasicSplittingSolver,
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


def main(N: int, dt: float, T: float, theta: float) -> Tuple[float, float, float, float]:
    """Run monodomain MMS."""
    mesh = UnitSquareMesh(N, N)
    time = Constant(0.0)
    lam = Constant(1.0)
    cell_model = NoCellModel()

    ac_str = "(8*pi*pi*lam*sin(t) + (lam + 1)*cos(t))*cos(2*pi*x[0])*cos(2*pi*x[1])/(lam + 1)"
    stimulus = Expression(ac_str, t=time, lam=lam, degree=3)
    brain = CardiacModel(mesh, time, 1.0, 1.0, cell_model, stimulus=stimulus)

    # Define solver solver
    ps = BasicSplittingSolver.default_parameters()
    ps["theta"] = theta
    ps["pde_solver"] = "monodomain"
    ps["BasicMonodomainSolver"]["linear_variational_solver"]["linear_solver"] = "direct"
    solver = BasicSplittingSolver(brain, params=ps)

    vs0 = Function(solver.VS)
    vs_, vs, vur = solver.solution_fields()
    vs_.assign(vs0)

    for timestep, (vs_, vs, vur) in solver.solve((0, T), dt):
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
        v_error, h, a, T = main(N*(2**level), a, T, theta)
        v_errors.append(v_error)
        dts.append(a)
        hs.append(h)

    v_rates = convergence_rate(hs, v_errors)
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
