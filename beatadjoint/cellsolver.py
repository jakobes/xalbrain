"This module contains solvers for (subclasses of) CardiacCellModel."

# Copyright (C) 2012 Marie E. Rognes (meg@simula.no)
# Use and modify at will
# Last changed: 2012-09-25

__all__ = ["CellSolver"]

from dolfin import *
from beatadjoint import CardiacCellModel

# ------------------------------------------------------------------------------
# Cardiac cell solver
# ------------------------------------------------------------------------------
class CellSolver:
    """This class provides a basic solver for cardiac cell models

    This solver solves the (nonlinear) ODE system described by the
    cell model using a basic theta-scheme. By default, theta=0.5,
    which corresponds to a Crank-Nicolson scheme.
    """

    def __init__(self, model, parameters=None):
        """Create solver from given CardiacCellModel (model) and
        optional parameters."""

        assert isinstance(model, CardiacCellModel), \
            "Expecting CardiacCellModel as first argument to CellSolver"

        # Set model and parameters
        self._model = model
        self.parameters = self.default_parameters()
        if parameters is not None:
            self.parameters.update(parameters)

        # Define domain (dummy, but carefully chosen)
        self._domain = UnitInterval(1)

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
        self.vs_ = Function(self.VS)
        self.vs = Function(self.VS)

    def default_parameters(self):
        "Set-up and return default parameters"
        parameters = Parameters("CellSolver")
        parameters.add("theta", 0.5)
        return parameters

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
            vs = self.step((t0, t1), self.vs_)
            self.vs.assign(vs)

            # Yield solutions
            yield (t0, t1), vs

            # Update members and move to next time
            self.vs_.assign(self.vs)
            t0 = t1
            t1 = t0 + dt

    def step(self, interval, ics):
        "Step through given interval with given initial conditions"

        # Extract time domain
        (t0, t1) = interval
        k_n = Constant(t1 - t0)

        # Extract initial conditions
        (v_, s_) = split(ics)

        # Set-up current variables
        vs = Function(self.VS)
        vs.assign(ics) # Start with good guess
        (v, s) = split(vs)
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
        if self._model.applied_current:
            t = t0 + theta*(t1 - t0)
            self._model.applied_current.t = t
            I_theta += self._model.applied_current

        # Set-up system
        G = (Dt_v - I_theta)*w*dx + inner(Dt_s - F_theta, r)*dx

        # Solve system
        pde = NonlinearVariationalProblem(G, vs, J=derivative(G, vs))
        solver = NonlinearVariationalSolver(pde)
        solver.solve()

        return vs
