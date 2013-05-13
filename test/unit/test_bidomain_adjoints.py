"""
Unit tests for various types of solvers for cardiac cell models.
"""

__author__ = "Marie E. Rognes (meg@simula.no), 2013"
__all__ = ["TestBasicBidomainSolverAdjoint",
           "TestBidomainSolverAdjoint"]

import unittest
from dolfin import *
from dolfin_adjoint import *
from beatadjoint import *

class TestCase(object):
    def __init__(self):
        self.mesh = UnitCubeMesh(5, 5, 5)
        self.time = Constant(0.0)

        # Create stimulus
        self.stimulus = Expression("2.0")

        # Create applied current
        self.applied_current = Expression("t", t=self.time)

        # Create conductivity "tensors"
        self.M_i = 1.0
        self.M_e = 2.0

        self.t0 = 0.0
        self.dt = 0.1
        self.T = 5*self.dt

class TestBasicBidomainSolverAdjoint(unittest.TestCase):
    "Test adjoint functionality for the basic bidomain solver."

    def setUp(self):
        adj_reset()
        self.case = TestCase()

    def test_replay(self):
        "Test that replay of basic bidomain solver reports success."
        Solver = BasicBidomainSolver
        case = self.case

        # Create solver
        params = Solver.default_parameters()
        params["linear_variational_solver"]["linear_solver"] = "lu"

        solver = Solver(case.mesh, case.time, case.M_i, case.M_e,
                        I_s=case.stimulus,
                        I_a=case.applied_current,
                        params=params)

        # Solve
        info_green("Running forward model")
        solutions = solver.solve((case.t0, case.t0 + case.T), case.dt)
        for (interval, fields) in solutions:
            (v_, vs) = fields

        # Check replay
        info_green("Running replay")
        success = replay_dolfin(stop=True, tol=0.0)
        self.assertEqual(success, True)

class TestBidomainSolverAdjoint(unittest.TestCase):
    "Test adjoint functionality for the bidomain solver."

    def setUp(self):
        adj_reset()
        self.case = TestCase()

    def test_replay(self):
        "Test that replay of basic bidomain solver reports success."
        Solver = BidomainSolver
        case = self.case

        # Create solver
        params = Solver.default_parameters()
        params["linear_solver_type"] = "direct"
        solver = Solver(case.mesh, case.time, case.M_i, case.M_e,
                        I_s=case.stimulus,
                        I_a=case.applied_current, params=params)

        # Solve
        info_green("Running forward model")
        solutions = solver.solve((case.t0, case.t0 + case.T), case.dt)
        for (interval, fields) in solutions:
            (v_, vs) = fields

        # Check replay
        info_green("Running replay")
        success = replay_dolfin(stop=True, tol=0.0)
        self.assertEqual(success, True)


if __name__ == "__main__":
    print("")
    print("Testing adjoints of bidomain solvers")
    print("------------------------------------")
    unittest.main()
