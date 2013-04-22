"""
Unit tests for various types of bidomain solver
"""

__author__ = "Marie E. Rognes (meg@simula.no), 2013"
__all__ = [""]

import unittest
from dolfin import *
from beatadjoint import *

class TestBasicBidomainSolver(unittest.TestCase):
    "Test functionality for the basic bidomain solver."

    def setUp(self):
        self.mesh = UnitCubeMesh(5, 5, 5)

        # Create stimulus
        self.stimulus = Expression("2.0")

        # Create ac
        self.applied_current = Expression("t", t=0)

        # Create conductivity "tensors"
        self.M_i = 1.0
        self.M_e = 2.0

        self.t0 = 0.0
        self.dt = 0.1

    def test_basic_solve(self):
        "Test that solver runs."

        Solver = BasicBidomainSolver

        # Create solver
        solver = Solver(self.mesh, self.M_i, self.M_e, I_s=self.stimulus,
                        I_a=self.applied_current)

        # Solve
        solutions = solver.solve((self.t0, self.t0 + 2*self.dt), self.dt)
        for (interval, fields) in solutions:
            (v_, vs) = fields

    def test_compare_solve_step(self):
        "Test that solve gives same results as single step"

        Solver = BasicBidomainSolver
        solver = Solver(self.mesh, self.M_i, self.M_e, I_s=self.stimulus,
                        I_a=self.applied_current)

        (v_, vs) = solver.solution_fields()

        # Solve
        interval = (self.t0, self.t0 + self.dt)
        solutions = solver.solve(interval, self.dt)
        for (interval, fields) in solutions:
            (v_, vs) = fields
            a = vs.vector().norm("l2")

        # Reset v_
        v_.vector()[:] = 0.0

        # Step
        solver.step(interval)
        b = vs.vector().norm("l2")

        # Check that result from solve and step match.
        self.assertEqual(a, b)


class TestBidomainSolver(unittest.TestCase):
    def setUp(self):
        self.mesh = UnitCubeMesh(5, 5, 5)

        # Create stimulus
        self.stimulus = Expression("2.0")

        # Create ac
        self.applied_current = Expression("t", t=0)

        # Create conductivity "tensors"
        self.M_i = 1.0
        self.M_e = 2.0

        self.t0 = 0.0
        self.dt = 0.1

    def test_basic_solve(self):
        "Test that solver runs."

        Solver = BidomainSolver

        # Create solver
        solver = Solver(self.mesh, self.M_i, self.M_e, I_s=self.stimulus,
                        I_a=self.applied_current)

        # Solve
        solutions = solver.solve((self.t0, self.t0 + 2*self.dt), self.dt)
        for (interval, fields) in solutions:
            (v_, vs) = fields

    def test_compare_with_basic_solve(self):
        "Test that solver gives same results as basic bidomain solver."

        # Create solver and solve
        solver = BidomainSolver(self.mesh, self.M_i, self.M_e,
                                I_s=self.stimulus,
                                I_a=self.applied_current)
        solutions = solver.solve((self.t0, self.t0 + 2*self.dt), self.dt)
        for (interval, fields) in solutions:
            (v_, vs) = fields
        bidomain_result = vs.vector().norm("l2")

        # Reset
        v_.vector()[:] = 0.0

        # Create solver and solve
        solver = BasicBidomainSolver(self.mesh, self.M_i, self.M_e,
                                     I_s=self.stimulus,
                                     I_a=self.applied_current)
        solutions = solver.solve((self.t0, self.t0 + 2*self.dt), self.dt)
        for (interval, fields) in solutions:
            (v_, vs) = fields
        basic_bidomain_result = vs.vector().norm("l2")

        self.assertEqual(bidomain_result, basic_bidomain_result)

if __name__ == "__main__":
    print("")
    print("Testing bidomain solvers")
    print("------------------------")
    unittest.main()
