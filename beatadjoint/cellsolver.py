"This module contains solvers for (subclasses of) CardiacCellModel."

__author__ = "Marie E. Rognes (meg@simula.no), 2012--2013"

__all__ = ["BasicSingleCellSolver"]

from dolfin import *
from dolfin_adjoint import *
from beatadjoint import CardiacCellModel
from beatadjoint.utils import state_space

class BasicSingleCellSolver:
    """A basic, non-optimised solver for standalone cardiac cell
    models.

    The nonlinear ODE system described by the cell model is solved via
    a theta-scheme.  By default theta=0.5, which corresponds to a
    Crank-Nicolson scheme. This can be changed by modifying the solver
    parameters.

    .. note::

       For the sake of simplicity and consistency with other solver
       objects, this solver operates on its solution fields (as state
       variables) directly internally. More precisely, solve (and
       step) calls will act by updating the internal solution
       fields. It implies that initial conditions can be set (and are
       intended to be set) by modifying the solution fields prior to
       simulation.

    *Arguments*
      model (:py:class:`~beatadjoint.cellmodels.cardiaccellmodel.CardiacCellModel`)
        The cardiac cell model in
      params (:py:class:`dolfin.Parameters`, optional)
        Solver parameters

    """

    def __init__(self, model, params=None):
        "Create solver from given cell model and optional parameters."

        assert isinstance(model, CardiacCellModel), \
            "Expecting CardiacCellModel as first argument to BasicSingleCellSolver"

        # Set model and parameters
        self._model = model
        self.parameters = self.default_parameters()
        if params is not None:
            self.parameters.update(params)

        # Define domain (carefully chosen dummy)
        self._domain = UnitIntervalMesh(1)

        # Extract number of state variables from model
        num_states = self._model.num_states()

        # Create (mixed) function space for potential + states
        V = FunctionSpace(self._domain, "DG", 0)
        S = state_space(self._domain, num_states, "DG", 0)
        self.VS = V*S

        # Initialize helper functions
        self.vs_ = Function(self.VS, name="vs_")
        self.vs = Function(self.VS, name="vs")

    @staticmethod
    def default_parameters():
        """Initialize and return a set of default parameters

        *Returns*
          A set of parameters (:py:class:`dolfin.Parameters`)
        """
        params = Parameters("BasicSingleCellSolver")
        params.add("theta", 0.5)
        return params

    def solution_fields(self):
        """
        Return tuple of previous and current solution objects.

        Modifying these will modify the solution objects of the solver
        and thus provides a way for setting initial conditions for
        instance.

        *Returns*
          (previous vs, current vs) (:py:class:`tuple` of :py:class:`dolfin.Function`)
        """
        return (self.vs_, self.vs)

    def solve(self, interval, dt=None):
        """
        Solve the problem given by the model on a given time interval
        (t0, t1) with a given timestep dt and return generator for a
        tuple of the interval and the current vs solution.

        *Arguments*
          interval (:py:class:`tuple`)
            The time interval for the solve given by (t0, t1)
          dt (int, optional)
            The timestep for the solve. Defaults to length of interval

        *Returns*
          (timestep, current vs) via (:py:class:`genexpr`)

        *Example of usage*::

          # Create generator
          solutions = solver.solve((0.0, 1.0), 0.1)

          # Iterate over generator (computes solutions as you go)
          for (interval, vs) in solutions:
            # do something with the solutions

        """

        # Solve on entire interval if no interval is given.
        if dt is None:
            dt = (T - T0)

        # FIXME: Add check that T >= T0 + dt

        # Initial set-up
        (T0, T) = interval
        t0 = T0
        t1 = T0 + dt

        # Step through time steps until at end time
        while (True) :
            info_blue("Solving on t = (%g, %g)" % (t0, t1))
            self._step((t0, t1))

            # Yield solutions
            yield (t0, t1), self.vs

            # Break if this is the last step
            if ((t1 + dt) > T):
                break

            # If not: update members and move to next time
            self.vs_.assign(self.vs)
            t0 = t1
            t1 = t0 + dt

    def _step(self, interval):
        """
        Step the solver forward on the given time interval (t0,
        t1). Users are recommended to use solve instead.

        *Arguments*
          interval (:py:class:`tuple`)
            The time interval for the solve given by (t0, t1)
        """

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
