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
            (v_, vur) = fields
            a = vur.vector().norm("l2")

        # Reset v_
        v_.vector()[:] = 0.0

        # Step
        solver.step(interval)
        b = vs.vector().norm("l2")

        # Check that result from solve and step match.
        self.assertEqual(a, b)


class TestBidomainSolver(unittest.TestCase):
    def setUp(self):
        N = 5
        self.mesh = UnitCubeMesh(N, N, N)

        # Create stimulus
        self.stimulus = Expression("2.0")

        # Create ac
        self.applied_current = Expression("t", t=0)

        # Create conductivity "tensors"
        self.M_i = 1.0
        self.M_e = 2.0

        self.t0 = 0.0
        self.dt = 0.1

    def test_solve(self):
        "Test that solver runs."

        # Create solver and solve
        solver = BidomainSolver(self.mesh, self.M_i, self.M_e,
                                I_s=self.stimulus,
                                I_a=self.applied_current)
        solutions = solver.solve((self.t0, self.t0 + 2*self.dt), self.dt)
        for (interval, fields) in solutions:
            (v_, vur) = fields

    def test_compare_with_basic_solve(self):
        """Test that solver with direct linear algebra gives same
        results as basic bidomain solver."""

        # Create solver and solve
        params = BidomainSolver.default_parameters()
        params["linear_solver_type"] = "direct"
        params["use_avg_u_constraint"] = True
        params["default_timestep"] = self.dt
        solver = BidomainSolver(self.mesh, self.M_i, self.M_e,
                                I_s=self.stimulus,
                                I_a=self.applied_current, params=params)
        solutions = solver.solve((self.t0, self.t0 + 2*self.dt), self.dt)
        for (interval, fields) in solutions:
            (v_, vur) = fields
        bidomain_result = vur.vector().norm("l2")

        # Create other solver and solve
        solver = BasicBidomainSolver(self.mesh, self.M_i, self.M_e,
                                     I_s=self.stimulus,
                                     I_a=self.applied_current)
        solutions = solver.solve((self.t0, self.t0 + 2*self.dt), self.dt)
        for (interval, fields) in solutions:
            (v_, vur) = fields
        basic_bidomain_result = vur.vector().norm("l2")

        print bidomain_result
        print basic_bidomain_result
        self.assertAlmostEqual(bidomain_result, basic_bidomain_result,
                               places=13)

    def test_compare_direct_iterative(self):
        "Test that direct and iterative solution give comparable results."

        # Create solver and solve
        params = BidomainSolver.default_parameters()
        params["linear_solver_type"] = "direct"
        params["use_avg_u_constraint"] = True
        params["default_timestep"] = self.dt
        solver = BidomainSolver(self.mesh, self.M_i, self.M_e,
                                I_s=self.stimulus,
                                I_a=self.applied_current,
                                params=params)
        solutions = solver.solve((self.t0, self.t0 + 3*self.dt), self.dt)
        for (interval, fields) in solutions:
            (v_, vur) = fields
            (v, u, r) = vur.split(deepcopy=True)
            a = v.vector().norm("l2")

        # Create solver and solve using iterative means
        params = BidomainSolver.default_parameters()
        params["default_timestep"] = self.dt
        params["krylov_solver"]["monitor_convergence"] = True
        solver = BidomainSolver(self.mesh, self.M_i, self.M_e,
                                I_s=self.stimulus,
                                I_a=self.applied_current,
                                params=params)
        solutions = solver.solve((self.t0, self.t0 + 3*self.dt), self.dt)
        for (interval, fields) in solutions:
            (v_, vu) = fields
            (v, u) = vu.split(deepcopy=True)
            b = v.vector().norm("l2")

        print "lu gives ", a
        print "krylov gives ", b
        self.assertAlmostEqual(a, b, places=4)

if __name__ == "__main__":
    print("")
    print("Testing bidomain solvers")
    print("------------------------")
    unittest.main()
