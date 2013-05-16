"""
Unit tests for various types of bidomain solver
"""

__author__ = "Marie E. Rognes (meg@simula.no), 2013"
__all__ = [""]

import unittest
from dolfin import *
from beatadjoint import *

class TestSplittingSolver(unittest.TestCase):
    "Test functionality for the splitting solvers."

    def setUp(self):
        self.mesh = UnitCubeMesh(5, 5, 5)

        # Create time
        self.time = Constant(0.0)

        # Create stimulus
        self.stimulus = Expression("2.0*t", t=self.time)

        # Create ac
        self.applied_current = Expression("t", t=self.time)

        # Create conductivity "tensors"
        self.M_i = 1.0
        self.M_e = 2.0

        self.cell_model = FitzHughNagumoManual()
        self.cardiac_model = CardiacModel(self.mesh, self.time,
                                          self.M_i, self.M_e,
                                          self.cell_model,
                                          self.stimulus,
                                          self.applied_current)

        self.t0 = 0.0
        self.dt = 0.1
        self.T = self.t0 + 5*self.dt
        self.ics = self.cell_model.initial_conditions()


    def test_basic_and_optimised_splitting_solver_exact(self):
        """Test that basic and optimised splitting solvers yield
        essentially equal results when configured identically."""

        # Create basic solver
        params = BasicSplittingSolver.default_parameters()
        params["BasicCardiacODESolver"]["S_polynomial_family"] = "CG"
        params["BasicCardiacODESolver"]["S_polynomial_degree"] = 1
        solver = BasicSplittingSolver(self.cardiac_model, params=params)

        (vs_, vs, vur) = solver.solution_fields()
        vs_.assign(self.ics)

        # Solve
        solutions = solver.solve((self.t0, self.T), self.dt)
        for (interval, fields) in solutions:
            (vs_, vs, vur) = fields
        a = vs.vector().norm("l2")
        c = vur.vector().norm("l2")
        self.assertAlmostEqual(interval[1], self.T)

        adj_reset()

        # Create optimised solver with direct solution algorithm
        params = SplittingSolver.default_parameters()
        params["BidomainSolver"]["linear_solver_type"] = "direct"
        params["BidomainSolver"]["use_avg_u_constraint"] = True
        solver = SplittingSolver(self.cardiac_model, params=params)

        (vs_, vs, vur) = solver.solution_fields()
        vs_.assign(self.ics)

        # Solve again
        solutions = solver.solve((self.t0, self.T), self.dt)
        for (interval, fields) in solutions:
            (vs_, vs, vur) = fields
        self.assertAlmostEqual(interval[1], self.T)
        b = vs.vector().norm("l2")
        d = vur.vector().norm("l2")

        print "a, b = ", a, b
        print "c, d = ", c, d
        print "a - b = ", (a - b)
        print "c - d = ", (c - d)

        # Compare results
        self.assertAlmostEqual(a, b, delta=1.e-10)
        self.assertAlmostEqual(c, d, delta=1.e-10)

    def _test_basic_and_optimised_splitting_solver_krylov(self):
        """Check that basic solver and optimised inexact splitting
        solver yield _similar_ results when same function spaces are
        used."""

        # Create basic solver
        adj_reset()
        params = BasicSplittingSolver.default_parameters()
        params["BasicCardiacODESolver"]["S_polynomial_family"] = "CG"
        params["BasicCardiacODESolver"]["S_polynomial_degree"] = 1
        solver = BasicSplittingSolver(self.cardiac_model, params=params)

        (vs_, vs, vur) = solver.solution_fields()
        vs_.assign(self.ics)

        # Solve
        solutions = solver.solve((self.t0, self.T), self.dt)
        for (interval, fields) in solutions:
            (vs_, vs, vur) = fields
        a = vs.vector().norm("l2")
        c = vur.split(deepcopy=True)[1].vector().norm("l2")

        # Create optimised solver
        adj_reset()
        solver = SplittingSolver(self.cardiac_model)

        (vs_, vs, vur) = solver.solution_fields()
        vs_.assign(self.ics)

        # Solve again
        solutions = solver.solve((self.t0, self.T), self.dt)
        for (interval, fields) in solutions:
            (vs_, vs, vu) = fields
        b = vs.vector().norm("l2")
        d = vu.split(deepcopy=True)[1].vector().norm("l2")

        # Evaluate difference in results
        print "a, b = ", a, b
        print "a - b = ", a - b
        print "c, d = ", c, d
        print "c - d = ", c - d

        # Expecting result in vs to be pretty equal
        self.assertAlmostEqual(a, b, delta=1.e-3)

        # Expecting result in u to be not quite equal
        self.assertAlmostEqual(c, d, delta=1.e-3)


if __name__ == "__main__":
    print("")
    print("Testing splitting solvers")
    print("------------------------")
    unittest.main()
