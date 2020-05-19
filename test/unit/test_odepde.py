"""Test of the operator splitting solver. Compare the solution of a
cell model over a single cell with a spatially and temporally constant
stimulus, with the solution of the bidomain equation with this cell
model and the same stimulus. We expect the solutions to be the same
modulo precision.
"""

__author__ = "Jakob Schreiner (jakob@simula.no), 2018"
__all__ = []

# Modified by Marie E. Rognes (meg@simula.no), 2018

import pytest
import itertools

import numpy as np

from xalbrain import (
    SingleCellSolver,
    Model,
    SplittingSolver,
)

import dolfin as df

from xalbrain.cellmodels import FitzHughNagumoManual

from testutils import assert_almost_equal


@pytest.mark.parametrize(
    ("theta", "pde_solver"),
    list(itertools.product([0.5, 1.0], ["monodomain", "bidomain"]))
)
def test_ode_pde(theta, pde_solver) -> None:
    """Test that the ode-pde coupling reproduces the ode solution."""
    parameters = SingleCellSolver.default_parameters()
    parameters["scheme"] = "RK4"
    time = df.Constant(0.0)
    stimulus = df.Expression("100", degree=1)
    model = FitzHughNagumoManual()
    model.stimulus = stimulus

    dt = 0.1
    T = 10*dt

    # Just propagate ODE solver
    solver = SingleCellSolver(time=time, cell_model=model, parameters=parameters)
    ode_vs_, ode_vs = solver.solution_fields()
    ode_vs_.assign(model.initial_conditions())
    for _ in solver.solve(0, T, dt):
        pass

    print("ODE")
    print("v(T) = ", ode_vs.vector().get_local()[0])
    print("s(T) = ", ode_vs.vector().get_local()[1])

    # Propagate with Bidomain+ODE solver
    mesh = df.UnitSquareMesh(1, 1)
    brain = Model(mesh, time, 1.0, 1.0, model, stimulus=stimulus)
    ps = SplittingSolver.default_parameters()
    ps["pde_solver"] = pde_solver
    ps["theta"] = float(theta)
    ps["CardiacODESolver"]["scheme"] = "RK4"
    ps["BidomainSolver"]["Chi"] = 1.0
    ps["BidomainSolver"]["Cm"] = 1.0
    solver = SplittingSolver(brain, parameters=ps)

    pde_vs_, pde_vs, vur = solver.solution_fields()
    pde_vs_.assign(model.initial_conditions())

    solutions = solver.solve(0, T, dt)
    for _ in solutions:
        pass

    n = mesh.num_vertices()
    print("PDE (%s, %g)" % (pde_solver, theta))
    print("v(T) = ", [pde_vs.vector().get_local()[2*i] for i in range(n)])
    print("s(T) = ", [pde_vs.vector().get_local()[2*i + 1] for i in range(n)])

    pde_vec = pde_vs.vector().get_local()
    ode_vec = ode_vs.vector().get_local()

    # Compare PDE and ODE solutions, we expect these to be essentially equal
    tolerance = 1e-3
    # assert_almost_equal(abs(pde_vec[0] - ode_vec[0]), 0.0, tolerance)
    # assert_almost_equal(abs(pde_vec[1] - ode_vec[1]), 0.0, tolerance)

    msg = pde_vec[:ode_vec.size], ode_vec
    assert np.allclose(pde_vec[:ode_vec.size], ode_vec, atol=tolerance), msg


if __name__ == "__main__":
    test_ode_pde(0.5, "bidomain")
