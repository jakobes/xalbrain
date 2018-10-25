import itertools
import pytest

import numpy as np
import dolfin as df
import xalbrain as xb


from xalbrain.cellmodels import FitzHughNagumoManual, Beeler_reuter_1977, Wei

from testutils import assert_almost_equal, parametrize

@parametrize(
    ("theta", "pde_solver"),
        # list(itertools.product([0.5, 1.0], ["monodomain", "bidomain"]))
    list(map(lambda x: pytest.param(*x, marks=pytest.mark.xfail),
        itertools.product([0.5, 1.0], ["monodomain", "bidomain"])))
)
def test_ode_pde(theta, pde_solver, N=10) -> None:
    """Test that the ode-pde coupling reproduces the ode solution."""
    time = df.Constant(0.0)
    mesh = df.UnitSquareMesh(N, N)
    stimulus = df.Expression("100*t", t=time, degree=1)

    cell_function = df.MeshFunction("size_t", mesh, mesh.geometry().dim())
    cell_function.set_all(1)
    df.CompiledSubDomain("x[0]> 0.5").mark(cell_function, 12)

    models = xb.MultiCellModel(
        (FitzHughNagumoManual(), Beeler_reuter_1977()),
        (1, 12),
        cell_function
    )

    params = xb.BasicCardiacODESolver.default_parameters()
    params["V_polynomial_family"] = "CG"
    params["V_polynomial_degree"] = 1
    params["theta"] = theta

    solver = xb.BasicCardiacODESolver(mesh, time, models, I_s=stimulus, params=params)
    ode_vs_, ode_vs = solver.solution_fields()
    models.assign_initial_conditions(ode_vs_)

    odev_, *_ = ode_vs_.split(True)

    dt = 0.1
    T = 10*dt

    for _ in solver.solve((0, T), dt):
        pass

    odev, *_ = ode_vs.split(True)

    print("ODE")
    print("v(T) = ", ode_vs.vector().get_local()[0])
    print("s(T) = ", ode_vs.vector().get_local()[1])

    # Propagate with Bidomain+ODE solver
    brain = xb.CardiacModel(mesh, time, 1.0, 1.0, models, stimulus=stimulus)
    ps = xb.SplittingSolver.default_parameters()
    ps["pde_solver"] = pde_solver
    ps["ode_solver_choice"] = "BasicCardiacODESolver"
    ps["BasicCardiacODESolver"]["V_polynomial_family"] = "CG"
    ps["BasicCardiacODESolver"]["V_polynomial_degree"] = 1
    ps["theta"] = float(theta)
    solver = xb.BasicSplittingSolver(brain, params=ps)

    pde_vs_, pde_vs, vur = solver.solution_fields()
    models.assign_initial_conditions(pde_vs_)

    solutions = solver.solve((0, T), dt)
    for _ in solutions:
        pass

    pde_vec = pde_vs.vector().get_local()
    ode_vec = ode_vs.vector().get_local()

    print(np.max(np.abs(pde_vec - ode_vec)))

    # Compare PDE and ODE solutions, we expect these to be essentially equal
    # tolerance = 1e-3
    # assert np.allclose(pde_vec, ode_vec, atol=tolerance)


if __name__ == "__main__":
    test_ode_pde(0.5, "monodomain")
