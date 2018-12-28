"""Unit tests for various types of bidomain solver."""

__author__ = "Marie E. Rognes (meg@simula.no), 2013"


from testutils import assert_almost_equal, assert_equal, fast

from dolfin import *

from xalbrain import (
    BasicBidomainSolver,
    BasicMonodomainSolver,
    MonodomainSolver,
    BidomainSolver,
)


class TestBasicBidomainSolver:
    """Test functionality for the basic bidomain solver."""

    def setUp(self) -> None:
        """Set up experiment."""
        self.mesh = UnitCubeMesh(5, 5, 5)
        self.time = Constant(0.0)

        # Create stimulus
        self.stimulus = Expression("2.0", degree=1)

        # Create ac
        self.applied_current = Expression(
            "sin(2*pi*x[0])*t",
            t=self.time,
            degree=3
        )

        # Create conductivity "tensors"
        self.M_i = 1.0
        self.M_e = 2.0

        self.t0 = 0.0
        self.dt = 0.1

    @fast
    def test_basic_solve(self) -> None:
        """Test that solver runs."""
        self.setUp()

        # Create solver
        solver = BasicBidomainSolver(
            self.mesh,
            self.time,
            self.M_i,
            self.M_e,
            I_s=self.stimulus,
            I_a=self.applied_current
        )

        # Solve
        solutions = solver.solve((self.t0, self.t0 + 2*self.dt), self.dt)
        for _, _ in solutions:
            pass

    @fast
    def test_compare_solve_step(self) -> None:
        "Test that solve gives same results as single step"
        self.setUp()

        solver = BasicBidomainSolver(
            self.mesh,
            self.time,
            self.M_i,
            self.M_e,
            I_s=self.stimulus,
            I_a=self.applied_current
        )

        v_, vs = solver.solution_fields()

        # Solve
        interval = (self.t0, self.t0 + self.dt)
        for _, (v_, vur) in solver.solve(interval, self.dt):
            a = vur.vector().norm("l2")

        # Reset v_
        v_.vector()[:] = 0.0

        # Step
        solver.step(interval)
        b = vs.vector().norm("l2")

        # Check that result from solve and step match.
        assert_equal(a, b)


class TestBasicMonodomainSolver:
    """Test functionality for the basic monodomain solver."""

    def setUp(self) -> None:
        self.mesh = UnitCubeMesh(5, 5, 5)
        self.time = Constant(0.0)

        # Create stimulus
        self.stimulus = Expression("2.0", degree=1)

        # Create conductivity "tensors"
        self.M_i = 1.0

        self.t0 = 0.0
        self.dt = 0.1

    @fast
    def test_basic_solve(self) -> None:
        """Test that solver runs."""
        self.setUp()

        # Create solver
        solver = BasicMonodomainSolver(
            self.mesh,
            self.time,
            self.M_i,
            I_s=self.stimulus
        )

        # Solve
        for _, _ in solver.solve((self.t0, self.t0 + 2*self.dt), self.dt):
            pass

    @fast
    def test_compare_solve_step(self) -> None:
        """Test that solve gives same results as single step."""
        self.setUp()

        solver = BasicMonodomainSolver(
            self.mesh,
            self.time,
            self.M_i,
            I_s=self.stimulus
        )

        v_, vs = solver.solution_fields()

        # Solve
        interval = (self.t0, self.t0 + self.dt)
        for _, (v_, vur) in solver.solve(interval, self.dt):
            a = vur.vector().norm("l2")

        # Reset v_
        v_.vector()[:] = 0.0

        # Step
        solver.step(interval)
        b = vs.vector().norm("l2")

        # Check that result from solve and step match.
        assert_equal(a, b)


class TestBidomainSolver:
    """Test functionality for the optimsed bidomain solver."""

    def setUp(self) -> None:
        N = 5
        self.mesh = UnitCubeMesh(N, N, N)
        self.time = Constant(0.0)

        # Create stimulus
        self.stimulus = Expression("2.0", degree=1)

        # Create ac
        self.applied_current = Expression(
            "sin(2*pi*x[0])*t",
            t=self.time,
            degree=3
        )

        # Create conductivity "tensors"
        self.M_i = 1.0
        self.M_e = 2.0

        self.t0 = 0.0
        self.dt = 0.1

    @fast
    def test_solve(self) -> None:
        """Test that solver runs."""
        self.setUp()

        # Create solver and solve
        solver = BidomainSolver(
            self.mesh,
            self.time,
            self.M_i,
            self.M_e,
            I_s=self.stimulus,
            I_a=self.applied_current
        )

        for _, _ in solver.solve((self.t0, self.t0 + 2*self.dt), self.dt):
            pass

    @fast
    def test_compare_with_basic_solve(self) -> None:
        """Test that direct solver gives same result as basic solver."""
        self.setUp()

        # Create solver and solve
        params = BidomainSolver.default_parameters()
        params["linear_solver_type"] = "direct"
        params["use_avg_u_constraint"] = True
        solver = BidomainSolver(
            self.mesh,
            self.time,
            self.M_i,
            self.M_e,
            I_s=self.stimulus,
            I_a=self.applied_current,
            params=params
        )

        for _, (v_, vur) in solver.solve((self.t0, self.t0 + 2*self.dt), self.dt):
            pass

        bidomain_result = vur.vector().norm("l2")

        params = BasicBidomainSolver.default_parameters()
        params["linear_solver_type"] = "direct"
        params["use_avg_u_constraint"] = True

        # Create other solver and solve
        solver = BasicBidomainSolver(
            self.mesh,
            self.time,
            self.M_i,
            self.M_e,
            I_s=self.stimulus,
            I_a=self.applied_current
        )

        for _, (v_, vur) in solver.solve((self.t0, self.t0 + 2*self.dt), self.dt):
            pass

        basic_bidomain_result = vur.vector().norm("l2")

        print(bidomain_result)
        print(basic_bidomain_result)
        assert_almost_equal(bidomain_result, basic_bidomain_result, 1e-13)

    @fast
    def test_compare_direct_iterative(self) -> None:
        """Test that direct and iterative solution give comparable results."""
        self.setUp()

        # Create solver and solve
        params = BidomainSolver.default_parameters()
        params["linear_solver_type"] = "direct"
        params["use_avg_u_constraint"] = False
        solver = BidomainSolver(
            self.mesh,
            self.time,
            self.M_i,
            self.M_e,
            I_s=self.stimulus,
            I_a=self.applied_current,
            params=params
        )

        for _, (v_, vur),  in solver.solve((self.t0, self.t0 + 3*self.dt), self.dt):
            v, *_ = vur.split(deepcopy=True)
            a = v.vector().norm("l2")

        # Create solver and solve using iterative means
        params = BidomainSolver.default_parameters()
        params["linear_solver_type"] = "iterative"
        params["use_avg_u_constraint"] =  False
        solver = BidomainSolver(
            self.mesh,
            self.time,
            self.M_i,
            self.M_e,
            I_s=self.stimulus,
            I_a=self.applied_current,
            params=params
        )

        for _, (v_, vur) in solver.solve((self.t0, self.t0 + 3*self.dt), self.dt):
            v, *_ = vur.split(deepcopy=True)
            b = v.vector().norm("l2")

        print("lu gives ", a)
        print("krylov gives ", b)
        assert_almost_equal(a, b, 1e-4)


class TestMonodomainSolver:
    """Test functionality for the optimsed monodomain solver."""

    def setUp(self) -> None:
        N = 5
        self.mesh = UnitCubeMesh(N, N, N)
        self.time = Constant(0.0)

        # Create stimulus
        self.stimulus = Expression("2.0", degree=1)

        # Create conductivity "tensors"
        self.M_i = 1.0

        self.t0 = 0.0
        self.dt = 0.1

    @fast
    def test_solve(self) -> None:
        """Test that solver runs."""
        self.setUp()

        # Create solver and solve
        solver = MonodomainSolver(self.mesh, self.time, self.M_i, I_s=self.stimulus)
        for _, _ in solver.solve((self.t0, self.t0 + 2*self.dt), self.dt):
            pass

    @fast
    def test_compare_with_basic_solve(self) -> None:
        """Test thant the optimised and non optimised solvers give the same answer."""
        self.setUp()

        # Create solver and solve
        params = MonodomainSolver.default_parameters()
        params["linear_solver_type"] = "direct"
        solver = MonodomainSolver(
            self.mesh,
            self.time,
            self.M_i,
            I_s=self.stimulus,
            params=params
        )

        for _, (v_, vur) in solver.solve((self.t0, self.t0 + 2*self.dt), self.dt):
            pass
        monodomain_result = vur.vector()

        # Create other solver and solve
        params = BasicMonodomainSolver.default_parameters()
        params["linear_solver_type"] = "direct"
        solver = BasicMonodomainSolver(
            self.mesh,
            self.time,
            self.M_i,
            I_s=self.stimulus,
            params=params
        )

        for _, (v_, vur) in solver.solve((self.t0, self.t0 + 2*self.dt), self.dt):
            pass
        basic_monodomain_result = vur.vector()

        # print("monodomain_result = ", monodomain_result.array())
        # print("basic_monodomain_result = ", basic_monodomain_result.array())
        assert_almost_equal(
            monodomain_result.norm("l2"),
            basic_monodomain_result.norm("l2"),
            1e-13
        )

    @fast
    def test_compare_direct_iterative(self) -> None:
        """Test that direct and iterative solution give comparable results."""
        self.setUp()

        # Create solver and solve
        params = MonodomainSolver.default_parameters()
        params["linear_solver_type"] = "direct"
        solver = MonodomainSolver(
            self.mesh,
            self.time,
            self.M_i,
            I_s=self.stimulus,
            params=params
        )

        for _, (v_, v) in solver.solve((self.t0, self.t0 + 3*self.dt), self.dt):
            pass
        a = v.vector().norm("l2")

        # Create solver and solve using iterative means
        params = MonodomainSolver.default_parameters()
        params["linear_solver_type"] = "iterative"
        # params["krylov_solver"]["monitor_convergence"] = True
        solver = MonodomainSolver(
            self.mesh,
            self.time,
            self.M_i,
            I_s=self.stimulus,
            params=params
        )

        for _, (v_, v) in solver.solve((self.t0, self.t0 + 3*self.dt), self.dt):
            pass
        b = v.vector().norm("l2")

        print("lu gives ", a)
        print("krylov gives ", b)
        assert_almost_equal(a, b, 1e-4)


if __name__ == "__main__":
    # tester = TestBidomainSolver()
    # tester.test_compare_direct_iterative()

    tester = TestMonodomainSolver()
    tester.test_solve()
