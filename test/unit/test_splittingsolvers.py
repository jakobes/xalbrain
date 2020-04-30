
"""Unit tests for various types of bidomain solver."""

__author__ = "Marie E. Rognes (meg@simula.no), 2013"
__all__ = ["TestSplittingSolver"]


from testutils import assert_almost_equal, medium, parametrize

from xalbrain import (
    Model,
    BasicSplittingSolver,
    SplittingSolver,
    BasicCardiacODESolver,
    FitzHughNagumoManual,
)

import dolfin as df

import pytest

df.set_log_level(100)


class TestSplittingSolver:
    """Test functionality for the splitting solvers."""

    def setup(self) -> None:
        self.mesh = df.UnitCubeMesh(5, 5, 5)

        # Create time
        self.time = df.Constant(0.0)

        # Create stimulus
        self.stimulus = df.Expression("2.0*t", t=self.time, degree=1)

        # Create ac
        self.applied_current = df.Expression("sin(2*pi*x[0])*t", t=self.time, degree=3)

        # Create conductivity "tensors"
        self.M_i = 1.0
        self.M_e = 2.0

        self.cell_model = FitzHughNagumoManual()
        self.cardiac_model = Model(
            self.mesh,
            self.time,
            self.M_i,
            self.M_e,
            self.cell_model,
            self.stimulus,
            self.applied_current
        )

        dt = 0.05
        self.t0 = 0.0
        self.dt = dt

        self.T = self.t0 + 5*dt
        self.ics = self.cell_model.initial_conditions()


    @medium
    @pytest.mark.parametrize("solver_type", [
        pytest.param("direct"),
        pytest.param("iterative", marks=pytest.mark.xfail)
    ])
    def test_basic_and_optimised_splitting_solver_exact(self, solver_type) -> None:
        """
        Test that the optimised and basic solvers yield similar results.
        """
        # Create basic solver
        parameters = BasicSplittingSolver.default_parameters()
        parameters["BasicBidomainSolver"]["linear_solver_type"] = solver_type
        parameters["theta"] = 0.5
        if solver_type == "direct":
            parameters["BasicBidomainSolver"]["use_avg_u_constraint"] = True
        else:
            parameters["BasicBidomainSolver"]["use_avg_u_constraint"] = False

        solver = BasicSplittingSolver(self.cardiac_model, parameters=parameters)

        vs_, vs, vur = solver.solution_fields()
        vs_.assign(self.ics)

        # Solve
        solutions = solver.solve(self.t0, self.T, self.dt)
        for (t0, t1), fields in solutions:
            vs_, vs, vur = fields

        foo = vs.vector()
        basic_vs = vs.vector().norm("l2")
        basic_vur = vur.vector().norm("l2")
        assert_almost_equal(t1, self.T, 1e-10)

        # Create optimised solver with direct solution algorithm
        parameters = SplittingSolver.default_parameters()
        parameters["BidomainSolver"]["linear_solver_type"] = solver_type
        # if solver_type == "direct":
        #     parameters["BidomainSolver"]["use_avg_u_constraint"] = True
        # else:
        #     parameters["BidomainSolver"]["use_avg_u_constraint"] = False
        solver = SplittingSolver(self.cardiac_model, parameters=parameters)

        vs_, vs, vur = solver.solution_fields()
        vs_.assign(self.ics)

        # Solve again
        solutions = solver.solve(self.t0, self.T, self.dt)
        for (t0, t1), fields in solutions:
            vs_, vs, vur = fields

        assert_almost_equal(t1, self.T, 1e-10)
        optimised_vs = vs.vector().norm("l2")
        optimised_vur = vur.vector().norm("l2")

        # Compare results, discrepancy is in difference in ODE solves.
        assert_almost_equal(optimised_vs, basic_vs, tolerance=1)
        assert_almost_equal(optimised_vur, basic_vur, tolerance=1)


if __name__ == "__main__":
    tss = TestSplittingSolver()
    tss.setup()

    for solver in ("direct",):
        tss.test_basic_and_optimised_splitting_solver_exact(solver)
