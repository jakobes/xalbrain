
"""Unit tests for various types of bidomain solver."""

__author__ = "Marie E. Rognes (meg@simula.no), 2013"
__all__ = ["TestSplittingSolver"]


from testutils import assert_almost_equal, medium, parametrize

from dolfin import info, set_log_level, WARNING

from xalbrain import (
    CardiacModel,
    BasicSplittingSolver,
    SplittingSolver,
    BasicCardiacODESolver,
    FitzHughNagumoManual,
)

from dolfin import (
    Constant,
    Expression,
    UnitCubeMesh,
    parameters,
)

import pytest


set_log_level(WARNING)


class TestSplittingSolver(object):
    """Test functionality for the splitting solvers."""

    def setup(self) -> None:
        self.mesh = UnitCubeMesh(5, 5, 5)

        # Create time
        self.time = Constant(0.0)

        # Create stimulus
        self.stimulus = Expression("2.0*t", t=self.time, degree=1)

        # Create ac
        self.applied_current = Expression("sin(2*pi*x[0])*t", t=self.time, degree=3)

        # Create conductivity "tensors"
        self.M_i = 1.0
        self.M_e = 2.0

        self.cell_model = FitzHughNagumoManual()
        self.cardiac_model = CardiacModel(
            self.mesh,
            self.time,
            self.M_i,
            self.M_e,
            self.cell_model,
            self.stimulus,
            self.applied_current
        )

        dt = 0.1
        self.t0 = 0.0
        self.dt = [(0.0, dt), (dt*2, dt/2), (dt*4, dt)]
        # Test using variable dt interval but using the same dt.

        self.T = self.t0 + 5*dt
        self.ics = self.cell_model.initial_conditions()


    @medium
    # @parametrize(("solver_type"), ["direct", "iterative"])
    @pytest.mark.parametrize("solver_type", [
        pytest.param("direct"),
        pytest.param("iterative", marks=pytest.mark.xfail)
    ])
    def test_basic_and_optimised_splitting_solver_exact(self, solver_type) -> None:
        """
        Test that the optimised and basic solvers yield similar results.
        """
        # Create basic solver
        params = BasicSplittingSolver.default_parameters()
        params["BasicCardiacODESolver"]["S_polynomial_family"] = "CG"
        params["BasicCardiacODESolver"]["S_polynomial_degree"] = 1
        params["BasicBidomainSolver"]["linear_solver_type"] = solver_type
        if solver_type == "direct":
            params["BasicBidomainSolver"]["use_avg_u_constraint"] = True
        solver = BasicSplittingSolver(self.cardiac_model, params=params)

        vs_, vs, vur = solver.solution_fields()
        vs_.assign(self.ics)

        # Solve
        solutions = solver.solve((self.t0, self.T), self.dt)
        for (t0, t1), fields in solutions:
            vs_, vs, vur = fields

        foo = vs.vector()
        basic_vs = vs.vector().norm("l2")
        basic_vur = vur.vector().norm("l2")
        assert_almost_equal(t1, self.T, 1e-10)

        # Create optimised solver with direct solution algorithm
        params = SplittingSolver.default_parameters()
        params["BidomainSolver"]["linear_solver_type"] = solver_type
        params["enable_adjoint"] = False
        if solver_type == "direct":
            params["BidomainSolver"]["use_avg_u_constraint"] = True
        solver = SplittingSolver(self.cardiac_model, params=params)

        vs_, vs, vur = solver.solution_fields()
        vs_.assign(self.ics)

        # Solve again
        solutions = solver.solve((self.t0, self.T), self.dt)
        for (t0, t1), fields in solutions:
            vs_, vs, vur = fields

        assert_almost_equal(t1, self.T, 1e-10)
        bar = vs.vector()
        optimised_vs = vs.vector().norm("l2")
        optimised_vur = vur.vector().norm("l2")

        # Compare results, discrepancy is in difference in ODE solves.
        assert_almost_equal(optimised_vs, basic_vs, tolerance=1)
        assert_almost_equal(optimised_vur, basic_vur, tolerance=1)


if __name__ == "__main__":
    tss = TestSplittingSolver()
    tss.setup()
    for solver in ("iterative", "direct"):
        foo, bar = tss.test_basic_and_optimised_splitting_solver_exact(solver)
