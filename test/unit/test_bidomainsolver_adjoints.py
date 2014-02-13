"""
Unit tests for various types of solvers for cardiac cell models.
"""

__author__ = "Marie E. Rognes (meg@simula.no), 2013"
__all__ = ["TestBasicBidomainSolverAdjoint",
           "TestBidomainSolverAdjoint"]

from testutils import assert_equal, fast, slow, \
        adjoint, parametrize, assert_greater

from beatadjoint.dolfinimport import info_green, info_red
from beatadjoint import BasicBidomainSolver, BidomainSolver, \
        UnitCubeMesh, Constant, Expression, inner, dx, dt, \
        assemble, parameters, InitialConditionParameter, \
        replay_dolfin, Functional, FINISH_TIME, \
        compute_gradient_tlm, compute_gradient, \
        taylor_test, Function


class TestCase(object):
    def setup(self):
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

class TestBasicBidomainSolverAdjoint(TestCase):
    """Test adjoint functionality for the basic bidomain solver."""

    def _setup(self, solver_type):
        """Creates the basic bidomain solver."""
        # Create solver
        Solver = BasicBidomainSolver
        params = Solver.default_parameters()
        params.linear_variational_solver.linear_solver = \
                        "gmres" if solver_type == "iterative" else "lu"
        params.linear_variational_solver.preconditioner = 'petsc_amg'
        self.solver = Solver(self.mesh, self.time, self.M_i, self.M_e,
                        I_s=self.stimulus,
                        I_a=self.applied_current,
                        params=params)

    def _solve(self, ics=None):
        """ Runs the forward model with the basic bidomain solver. """
        print("Running forward basic model")

        (vs_, vs) = self.solver.solution_fields()
        solutions = self.solver.solve((self.t0, self.t0 + self.T), self.dt)

        # Set initial conditions
        if ics is not None:
            vs_.assign(ics)

        # Solve
        for (interval, fields) in solutions:
            pass

        return vs

    @adjoint
    @fast
    @parametrize("solver_type", ["iterative", "direct"])
    def test_replay(self, solver_type):
        "Test that replay of basic bidomain solver reports success."
        self._setup(solver_type)
        self._solve()

        # Check replay
        info_green("Running replay basic (%s)" % solver_type)
        success = replay_dolfin(stop=True, tol=0.0)
        assert_equal(success, True)

    def tlm_adj_setup(self, solver_type):
        """ Common code for test_tlm and test_adjoint. """
        self._setup(solver_type)
        self._solve()

        # Define functional
        (vs_, vs) = self.solver.solution_fields()
        form = lambda w: inner(w, w)*dx
        J = Functional(form(vs)*dt[FINISH_TIME])
        m = InitialConditionParameter(vs_)

        # Compute value of functional with current ics
        Jics = assemble(form(vs))

        # Define reduced functional
        def Jhat(ics):
            self._setup(solver_type)
            vs = self._solve(ics)
            return assemble(form(vs))

        # Stop annotating
        parameters["adjoint"]["stop_annotating"] = True

        return J, Jhat, m, Jics


    @adjoint
    @slow
    @parametrize("solver_type", ["iterative", "direct"])
    def test_tlm(self, solver_type):
        """Test that tangent linear model of basic bidomain solver converges at 2nd order."""
        info_green("Running tlm basic (%s)" % solver_type)

        J, Jhat, m, Jics = self.tlm_adj_setup(solver_type)

        # Check TLM correctness
        dJdics = compute_gradient_tlm(J, m, forget=False)
        assert (dJdics is not None), "Gradient is None (#fail)."
        conv_rate_tlm = taylor_test(Jhat, m, Jics, dJdics)

        # Check that minimal convergence rate is greater than some given number
        assert_greater(conv_rate_tlm, 1.9)


    @adjoint
    @slow
    @parametrize("solver_type", ["iterative", "direct"])
    def test_adjoint(self, solver_type):
        """Test that adjoint model of basic bidomain solver converges at 2nd order."""
        info_green("Running adjoint basic (%s)" % solver_type)

        J, Jhat, m, Jics = self.tlm_adj_setup(solver_type)

        # Check adjoint correctness
        dJdics = compute_gradient(J, m, forget=False)
        assert (dJdics is not None), "Gradient is None (#fail)."
        conv_rate = taylor_test(Jhat, m, Jics, dJdics)

        # Check that minimal convergence rate is greater than some given number
        assert_greater(conv_rate, 1.9)


class TestBidomainSolverAdjoint(TestCase):
    "Test adjoint functionality for the bidomain solver."

    def _run_replay(self, solver_type):
        "Test that replay of bidomain solver reports success."
        Solver = BidomainSolver

        # Create solver
        params = Solver.default_parameters()
        params.linear_solver_type = solver_type
        
        solver = Solver(self.mesh, self.time, self.M_i, self.M_e,
                        I_s=self.stimulus,
                        I_a=self.applied_current, params=params)

        # Solve
        info_green("Running forward model (%s)" % solver_type)
        solutions = solver.solve((self.t0, self.t0 + self.T), self.dt)
        for (interval, fields) in solutions:
            (v_, vs) = fields

        # Check replay
        if solver_type == "iterative":
            # FIXME: Bug in dolfin/dolfin_adjoint?
            return 
        info_green("Running replay (%s)" % solver_type)
        success = replay_dolfin(stop=True, tol=1.e-14)
        assert_equal(success, True)

    @adjoint
    @fast
    def test_replay_iterative(self):
        self.setUp()
        self._run_replay("iterative")

    @adjoint
    @fast
    def test_replay_direct(self):
        self.setUp()
        self._run_replay("direct")
