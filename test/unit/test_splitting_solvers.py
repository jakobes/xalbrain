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

        # Create stimulus
        self.stimulus = Expression("2.0")

        # Create ac
        self.applied_current = Expression("t", t=0)

        # Create conductivity "tensors"
        self.M_i = 1.0
        self.M_e = 2.0

        self.cell_model = NoCellModel()

        self.cardiac_model = CardiacModel(self.mesh, self.M_i, self.M_e,
                                          self.cell_model,
                                          self.stimulus,
                                          self.applied_current)

        self.t0 = 0.0
        self.dt = 0.1
        self.T = self.t0 + 5*self.dt

    def test_basic_and_optimised_splitting_solver(self):
        "Test that basic and optimised splitting solvers yield similar results."

        # Create basic solver
        solver = BasicSplittingSolver(self.cardiac_model)

        # Solve
        solutions = solver.solve((self.t0, self.T), self.dt)
        for (interval, fields) in solutions:
            (v_, vs, vur) = fields
        a = vs.vector().norm("l2")
        c = vur.vector().norm("l2")
        self.assertAlmostEqual(interval[1], self.T)

        # Create optimised solver with direct solution algorithm
        params = SplittingSolver.default_parameters()
        params["BidomainSolver"]["linear_solver_type"] = "direct"
        params["BidomainSolver"]["use_avg_u_constraint"] = True
        solver = SplittingSolver(self.cardiac_model, params=params)

        # Solve again
        solutions = solver.solve((self.t0, self.T), self.dt)
        for (interval, fields) in solutions:
            (v_, vs, vur) = fields
        self.assertAlmostEqual(interval[1], self.T)
        b = vs.vector().norm("l2")
        d = vur.vector().norm("l2")

        print "a, b = ", a, b
        print "c, d = ", a, b

        # Compare results
        self.assertAlmostEqual(a, b, places=6)
        self.assertAlmostEqual(c, d, places=6)

    def test_basic_and_optimised_splitting_solver2(self):
        "Check that optimised inexact splitting solver yield similar results."

        # Create basic solver
        solver = BasicSplittingSolver(self.cardiac_model)

        # Solve
        solutions = solver.solve((self.t0, self.T), self.dt)
        for (interval, fields) in solutions:
            (v_, vs, vur) = fields
        a = vs.vector().norm("l2")
        c = vur.split(deepcopy=True)[1].vector().norm("l2")

        # Create optimised solver
        solver = SplittingSolver(self.cardiac_model)

        # Solve again
        solutions = solver.solve((self.t0, self.T), self.dt)
        for (interval, fields) in solutions:
            (v_, vs, vu) = fields
        b = vs.vector().norm("l2")
        d = vu.split(deepcopy=True)[1].vector().norm("l2")

        print "a, b = ", a, b
        print "a - b = ", a - b

        # Expecting result in v to be pretty equal
        self.assertAlmostEqual(a, b, places=4)

        # Expecting result in u to be not so equal
        self.assertAlmostEqual(c, d, delta=1.e-2)


if __name__ == "__main__":
    print("")
    print("Testing splitting solvers")
    print("------------------------")
    unittest.main()
