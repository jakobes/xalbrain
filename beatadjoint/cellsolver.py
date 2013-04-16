"This module contains solvers for (subclasses of) CardiacCellModel."

__author__ = "Marie E. Rognes (meg@simula.no), 2012--2013"

__all__ = ["BasicSingleCellSolver"]

from dolfin import *
from dolfin_adjoint import *
from beatadjoint import CardiacCellModel

class BasicSingleCellSolver:
    """This class provides a basic solver for cardiac cell models.

    This solver solves the (nonlinear) ODE system described by the
    cell model using a basic theta-scheme. By default, theta=0.5,
    which corresponds to a Crank-Nicolson scheme.
    """

    def __init__(self, model, params=None):
        """Create solver from given CardiacCellModel (model) and
        optional parameters."""

        assert isinstance(model, CardiacCellModel), \
            "Expecting CardiacCellModel as first argument to BasicSingleCellSolver"

        # Set model and parameters
        self._model = model
        self.parameters = self.default_parameters()
        if params is not None:
            self.parameters.update(params)

        # Define domain (dummy, but carefully chosen)
        self._domain = UnitIntervalMesh(1)

        # Extract number of state variables from model
        num_states = self._model.num_states()

        # Create (mixed) function space for potential + states
        V = FunctionSpace(self._domain, "DG", 0)
        if num_states > 1:
            S = VectorFunctionSpace(self._domain, "DG", 0, num_states)
        else:
            S = FunctionSpace(self._domain, "DG", 0)
        self.VS = V*S

        # Initialize helper functions
        self.vs_ = Function(self.VS, name="vs_")
        self.vs = Function(self.VS, name="vs")

    @staticmethod
    def default_parameters():
        "Set-up and return default parameters"
        params = Parameters("BasicSingleCellSolver")
        params.add("theta", 0.5)
        return params

    def solution_fields(self):
        "Return tuple of 'previous' and 'current' solution fields."
        return (self.vs_, self.vs)

    def solve(self, interval, dt):
        """
        Return generator for solutions on given time interval (t0, t1)
        with given timestep 'dt'.
        """
        # Initial set-up
        (T0, T) = interval
        t0 = T0
        t1 = T0 + dt

        # Step through time steps until at end time
        while (t1 <= T) :
            info_blue("Solving on t = (%g, %g)" % (t0, t1))
            self.step((t0, t1))

            # Yield solutions
            yield (t0, t1), self.vs

            # Update members and move to next time
            self.vs_.assign(self.vs)
            t0 = t1
            t1 = t0 + dt

    def step(self, interval):
        "Step through given interval"

        # Extract time domain
        (t0, t1) = interval
        k_n = Constant(t1 - t0)

        # Extract previous solution(s)
        (v_, s_) = split(self.vs_)

        # Set-up current variables
        self.vs.assign(self.vs_) # Start with good guess
        (v, s) = split(self.vs)
        (w, r) = TestFunctions(self.VS)

        # Define equation based on cell model
        Dt_v = (v - v_)/k_n
        Dt_s = (s - s_)/k_n

        theta = self.parameters["theta"]
        F = self._model.F
        I_ion = self._model.I

        # Note sign for I_theta
        F_theta = theta*F(v, s) + (1 - theta)*F(v_, s_)
        I_theta = - (theta*(I_ion(v, s) + (1 - theta)*I_ion(v_, s_)))

        # Add current if applicable
        if self._model.stimulus:
            t = t0 + theta*(t1 - t0)
            self._model.stimulus.t = t
            I_theta += self._model.stimulus

        # Set-up system
        G = (Dt_v - I_theta)*w*dx + inner(Dt_s - F_theta, r)*dx

        # Solve system
        pde = NonlinearVariationalProblem(G, self.vs, J=derivative(G, self.vs))
        solver = NonlinearVariationalSolver(pde)
        solver.solve()
