"""
Unit tests for the merger in splitting solver
"""

__author__ = "Marie E. Rognes (meg@simula.no), 2013"
__all__ = ["TestMerger"]


import pytest

import numpy as np

from xalbrain import (
    Model,
    BasicSplittingSolver,
    SplittingSolver,
    FitzHughNagumoManual,
)

from dolfin import (
    UnitCubeMesh,
    Constant
)


class TestMerger:
    """Test functionality for the splitting solvers."""

    def setup(self):
        self.mesh = UnitCubeMesh(2, 2, 2)
        self.cell_model = FitzHughNagumoManual()
        self.cardiac_model = Model(self.mesh, None, 1.0, 2.0, self.cell_model)

    @pytest.mark.parametrize("Solver", [SplittingSolver, BasicSplittingSolver])
    def test_basic_and_optimised_splitting_solver_merge(self, Solver):
        """Test that the merger in basic and optimised splitting solver works."""

        # Create basic solver
        ps = Solver.default_parameters()
        if Solver is SplittingSolver:
            ps["BidomainSolver"]["Chi"] = 1.0
            ps["BidomainSolver"]["Cm"] = 1.0
        else:
            ps["BasicBidomainSolver"]["Chi"] = 1.0
            ps["BasicBidomainSolver"]["Cm"] = 1.0
        solver = Solver(self.cardiac_model, parameters=ps)

        vs_, vs, vur = solver.solution_fields()

        vs.vector()[:] = 2.0
        vur.vector()[:] = 1.0
        solver.merge(vs)

        tol = 1e-13
        assert np.abs(vs.sub(0, deepcopy=1).vector().get_local() - 1.0).max() < tol
        assert np.abs(vs.sub(1, deepcopy=1).vector().get_local() - 2.0).max() < tol
