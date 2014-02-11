"""
Unit tests for various types of solvers for cardiac cell models.
"""

__author__ = "Marie E. Rognes (meg@simula.no), 2013"
__all__ = ["TestBasicBidomainSolverAdjoint",
           "TestBidomainSolverAdjoint"]

from testutils import assert_equal

from beatadjoint.dolfinimport import info_green
from beatadjoint import BasicBidomainSolver, BidomainSolver, \
        UnitCubeMesh, Constant, Expression, \
        adj_reset, replay_dolfin


class TestCase(object):
    def __init__(self):
        self.mesh = UnitCubeMesh(5, 5, 5)
        self.time = Constant(0.0)

        # Create stimulus
        self.stimulus = {0:Expression("2.0")}

        # Create applied current
        self.applied_current = Expression("sin(2*pi*x[0])*t", t=self.time)

        # Create conductivity "tensors"
        self.M_i = 1.0
        self.M_e = 2.0

        self.t0 = 0.0
        self.dt = 0.1
        self.T = 5*self.dt

class TestBasicBidomainSolverAdjoint(object):
    "Test adjoint functionality for the basic bidomain solver."

    def setUp(self):
        adj_reset()
        self.case = TestCase()

    def _run_replay(self, solver_type):
        "Test that replay of basic bidomain solver reports success."
        Solver = BasicBidomainSolver
        case = self.case

        # Create solver
        params = Solver.default_parameters()
        params.linear_variational_solver.linear_solver = \
                        "gmres" if solver_type == "iterative" else "lu"
        solver = Solver(case.mesh, case.time, case.M_i, case.M_e,
                        I_s=case.stimulus,
                        I_a=case.applied_current,
                        params=params)

        # Solve
        info_green("Running forward basic model (%s)" % solver_type)
        solutions = solver.solve((case.t0, case.t0 + case.T), case.dt)
        for (interval, fields) in solutions:
            (v_, vs) = fields

        # Check replay
        info_green("Running replay basic (%s)" % solver_type)
        success = replay_dolfin(stop=True, tol=0.0)
        assert_equal(success, True)

    def test_replay_iterative(self):
        self.setUp()
        self._run_replay("iterative")

    def test_replay_direct(self):
        self.setUp()
        self._run_replay("direct")

class TestBidomainSolverAdjoint(object):
    "Test adjoint functionality for the bidomain solver."

    def setUp(self):
        adj_reset()
        self.case = TestCase()

    def _run_replay(self, solver_type):
        "Test that replay of bidomain solver reports success."
        Solver = BidomainSolver
        case = self.case

        # Create solver
        params = Solver.default_parameters()
        params.linear_solver_type = solver_type
        
        solver = Solver(case.mesh, case.time, case.M_i, case.M_e,
                        I_s=case.stimulus,
                        I_a=case.applied_current, params=params)

        # Solve
        info_green("Running forward model (%s)" % solver_type)
        solutions = solver.solve((case.t0, case.t0 + case.T), case.dt)
        for (interval, fields) in solutions:
            (v_, vs) = fields

        # Check replay
        if solver_type == "iterative":
            # FIXME: Bug in dolfin/dolfin_adjoint?
            return 
        info_green("Running replay (%s)" % solver_type)
        success = replay_dolfin(stop=True, tol=1.e-14)
        assert_equal(success, True)

    def test_replay_iterative(self):
        self.setUp()
        self._run_replay("iterative")

    def test_replay_direct(self):
        self.setUp()
        self._run_replay("direct")
