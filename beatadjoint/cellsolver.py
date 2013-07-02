"This module contains solvers for (subclasses of) CardiacCellModel."

__author__ = "Marie E. Rognes (meg@simula.no), 2012--2013"

__all__ = ["BasicSingleCellSolver",
           "BasicCardiacODESolver",
           "CardiacODESolver"]

from dolfin import *
from dolfin_adjoint import *
from beatadjoint import CardiacCellModel
from beatadjoint.utils import state_space, end_of_time

class BasicCardiacODESolver(object):
    """A basic, non-optimised solver for systems of ODEs typically
    encountered in cardiac applications of the form: find a scalar
    field :math:`v = v(x, t)` and a vector field :math:`s = s(x, t)`

    .. math::

      v_t = - I_{ion}(v, s) + I_s

      s_t = F(v, s)

    where :math:`I_{ion}` and :math:`F` are given non-linear
    functions, :math:`I_s` is some prescribed stimulus.

    Here, this nonlinear ODE system is solved via a theta-scheme.  By
    default theta=0.5, which corresponds to a Crank-Nicolson
    scheme. This can be changed by modifying the solver parameters.

    .. note::

       For the sake of simplicity and consistency with other solver
       objects, this solver operates on its solution fields (as state
       variables) directly internally. More precisely, solve (and
       step) calls will act by updating the internal solution
       fields. It implies that initial conditions can be set (and are
       intended to be set) by modifying the solution fields prior to
       simulation.

    *Arguments*
      domain (:py:class:`dolfin.Mesh`)
        The spatial domain (mesh)

      time (:py:class:`dolfin.Constant` or None)
        A constant holding the current time. If None is given, time is
        created for you, initialized to zero.

      num_states (int)
        The number of state variables (length of s)

      F (:py:func:`lambda`)
        A (non-)linear lambda vector function describing the evolution
        of the state variables (s)

      I_ion (:py:func:`lambda`)
        A (non-)linear lambda scalar function describing the evolution
        of the variable v

      I_s (:py:class:`dict`, optional)
        A typically time-dependent external stimulus given as a dict,
        with domain markers as the key and a
        :py:class:`dolfin.Expression` as values. NB: it is assumed
        that the time dependence of I_s is encoded via the 'time'
        Constant. 

      params (:py:class:`dolfin.Parameters`, optional)
        Solver parameters

    """
    def __init__(self, domain, time, num_states, F, I_ion,
                 I_s=None, params=None):

        # Store input
        self._domain = domain
        self._num_states = num_states
        self._F = F
        self._I_ion = I_ion
        self._I_s = I_s or {}

        assert isinstance(self._I_s, dict), "expects a dict with domain markers as "\
               "keys and stimulus expressions as values"

        # Check for stored mesh domains
        self._sub_domains = {}
        dim = domain.topology().dim()
        for domain_dim in [0, dim]:
            if domain.domains().num_marked(domain_dim):
                self._sub_domains[domain_dim] = MeshFunction(\
                    "size_t", domain, dim, domain.domains())

        # Create time if not given, otherwise use given time
        if time is None:
            self._time = Constant(0.0)
        else:
            self._time = time

        # Initialize and update parameters if given
        self.parameters = self.default_parameters()
        if params is not None:
            self.parameters.update(params)

        # Create (mixed) function space for potential + states
        family = self.parameters["V_polynomial_family"]
        degree = self.parameters["V_polynomial_degree"]
        V = FunctionSpace(self._domain, family, degree)
        family = self.parameters["S_polynomial_family"]
        degree = self.parameters["S_polynomial_degree"]
        S = state_space(self._domain, self._num_states, family, degree)
        self.VS = V*S

        # Initialize solution fields
        self.vs_ = Function(self.VS, name="vs_")
        self.vs = Function(self.VS, name="vs")

    @property
    def time(self):
        "The internal time of the solver."
        return self._time

    @staticmethod
    def default_parameters():
        """Initialize and return a set of default parameters

        *Returns*
          A set of parameters (:py:class:`dolfin.Parameters`)
        """
        params = Parameters("BasicCardiacODESolver")
        params.add("theta", 0.5)
        params.add("V_polynomial_degree", 0)
        params.add("V_polynomial_family", "DG")
        params.add("S_polynomial_degree", 0)
        params.add("S_polynomial_family", "DG")

        # Use iterative solver as default.
        params.add(NonlinearVariationalSolver.default_parameters())
        params["nonlinear_variational_solver"]["linear_solver"] = "gmres"

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

        # Initial time set-up
        (T0, T) = interval

        # Solve on entire interval if no interval is given.
        if dt is None:
            dt = (T - T0)

        t0 = T0
        t1 = T0 + dt

        # Check that we are not at end of time already.
        if end_of_time(T, None, t0, dt):
            info_red("Given end time %g is less than given increment %g", T, dt)

        # Step through time steps until at end time
        while (True) :
            self.step((t0, t1))

            # Yield solutions
            yield (t0, t1), self.vs

            # Break if this is the last step
            if end_of_time(T, t0, t1, dt):
                break

            # If not: update members and move to next time
            self.vs_.assign(self.vs)
            t0 = t1
            t1 = t0 + dt

    def step(self, interval):
        """
        Solve on the given time step (t0, t1).

        End users are recommended to use solve instead.

        *Arguments*
          interval (:py:class:`tuple`)
            The time interval (t0, t1) for the step
        """

        # Check for cell domains
        dim = self._domain.topology().dim()
        dxx = dx
        if dim in self._sub_domains:
            dxx = dxx[self._sub_domains[dim]]

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

        # Set time (propagates to time-dependent variables defined via
        # self.time)
        t = t0 + theta*(t1 - t0)
        self.time.assign(t)

        v_mid = theta*v + (1.0 - theta)*v_
        s_mid = theta*s + (1.0 - theta)*s_

        # Note sign for I_theta
        F_theta = self._F(v_mid, s_mid, time=self.time)
        I_theta = - self._I_ion(v_mid, s_mid, time=self.time)
        ## Note that if we only keep one time, then we cannot really use
        ## the formulation below.
        #F_theta = theta*self._F(v, s) + (1 - theta)*self._F(v_, s_)
        #I_theta = - (theta*self._I_ion(v, s) + (1 - theta)*self._I_ion(v_, s_))

        # Set-up system of equations
        G = (Dt_v - I_theta)*w*dx + inner(Dt_s - F_theta, r)*dx

        # Add stimulus (I_s) if applicable.
        # FIXME: domain == 0 the whole domain. Any better way to indicate a stimulus
        # FIXME: for the whole domain?
        for domain, I in self._I_s.items():
            if domain == 0:
                G -= I*w*dxx()
            else:
                G -= I*w*dxx(domain)

        # Solve system
        pde = NonlinearVariationalProblem(G, self.vs, J=derivative(G, self.vs))
        solver = NonlinearVariationalSolver(pde)
        solver_params = self.parameters["nonlinear_variational_solver"]
        solver.parameters.update(solver_params)
        solver.solve()

class CardiacODESolver(object):
    """An optimised solver for systems of ODEs typically
    encountered in cardiac applications of the form: find a scalar
    field :math:`v = v(x, t)` and a vector field :math:`s = s(x, t)`

    .. math::

      v_t = - I_{ion}(v, s) + I_s

      s_t = F(v, s)

    where :math:`I_{ion}` and :math:`F` are given non-linear
    functions, :math:`I_s` is some prescribed stimulus.

    .. note::

       For the sake of simplicity and consistency with other solver
       objects, this solver operates on its solution fields (as state
       variables) directly internally. More precisely, solve (and
       step) calls will act by updating the internal solution
       fields. It implies that initial conditions can be set (and are
       intended to be set) by modifying the solution fields prior to
       simulation.

    *Arguments*
      domain (:py:class:`dolfin.Mesh`)
        The spatial domain (mesh)

      time (:py:class:`dolfin.Constant` or None)
        A constant holding the current time. If None is given, time is
        created for you, initialized to zero.

      num_states (int)
        The number of state variables (length of s)

      F (:py:func:`lambda`)
        A (non-)linear lambda vector function describing the evolution
        of the state variables (s)

      I_ion (:py:func:`lambda`)
        A (non-)linear lambda scalar function describing the evolution
        of the variable v

      I_s (:py:class:`dolfin.Expression`, optional)
        A typically time-dependent external stimulus. NB: it is
        assumed that the time dependence of I_s is encoded via the
        'time' Constant.

      params (:py:class:`dolfin.Parameters`, optional)
        Solver parameters

    """
    def __init__(self, domain, time, num_states, F, I_ion,
                 I_s=None, params=None):

        # Store input
        self._domain = domain
        self._num_states = num_states
        self._F = F
        self._I_ion = I_ion
        self._I_s = I_s or {}

        # Create time if not given, otherwise use given time
        if time is None:
            self._time = Constant(0.0)
        else:
            self._time = time

        # Initialize and update parameters if given
        self.parameters = self.default_parameters()
        if params is not None:
            self.parameters.update(params)

        # Create (mixed) function space for potential + states
        V = FunctionSpace(self._domain, "CG", 1)
        S = state_space(self._domain, self._num_states)
        self.VS = V*S

        # Initialize solution field
        self.vs_ = Function(self.VS, name="vs_")
        self.vs = Function(self.VS, name="vs")

        # Initialize scheme
        (v, s) = split(self.vs)
        (w, q) = TestFunctions(self.VS)
        self._rhs = (inner(self._F(v, s, self._time), q)
                     - inner(self._I_ion(v, s, self._time), w))*dP

        # Add stimuli current
        assert len(self._I_s) in [0,1], "Domains are not supported for "\
               "PointIntegralSolver so we only accept one domain"
        for I in self._I_s.values():
            self._rhs += inner(I, w)*dP

        name = self.parameters["scheme"]
        Scheme = self._name_to_scheme(name)
        self._scheme = Scheme(self._rhs, self.vs, self._time)

        # Initialize solver and update its parameters
        self._pi_solver = PointIntegralSolver(self._scheme)
        self._pi_solver.parameters.update(self.parameters["point_integral_solver"])

    def _name_to_scheme(self, name):
        """Return scheme class with given name

        *Arguments*
          name (string)

        *Returns*
          the Scheme (:py:class:`dolfin.MultiStageScheme`)

        """
        return eval(name)

    @staticmethod
    def default_parameters():
        """Initialize and return a set of default parameters

        *Returns*
          A set of parameters (:py:class:`dolfin.Parameters`)
        """
        params = Parameters("CardiacODESolver")
        params.add("scheme", "BackwardEuler")
        params.add(PointIntegralSolver.default_parameters())

        return params

    def solution_fields(self):
        """
        Return current solution object.

        Modifying this will modify the solution object of the solver
        and thus provides a way for setting initial conditions for
        instance.

        *Returns*
          (previous vs_, current vs) (:py:class:`dolfin.Function`)
        """
        return (self.vs_, self.vs)

    def step(self, interval):
        """
        Solve on the given time step (t0, t1).

        End users are recommended to use solve instead.

        *Arguments*
          interval (:py:class:`tuple`)
            The time interval (t0, t1) for the step
        """
        # NB: The point integral solver operates on vs directly, map
        # initial condition in vs_ to vs:
        
        # FIXME: Shaky peformance in parallel?
        self.vs.assign(self.vs_)

        (t0, t1) = interval
        dt = t1 - t0
        self._pi_solver.step(dt)

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
        
        # Initial time set-up
        (T0, T) = interval

        # Solve on entire interval if no interval is given.
        if dt is None:
            dt = (T - T0)
        t0 = T0
        t1 = T0 + dt

        # Check that we are not at end of time already.
        if end_of_time(T, None, t0, dt):
            info_red("Given end time %g is less than given increment %g", T, dt)

        # Step through time steps until at end time
        while (True) :
            self.step((t0, t1))

            # Yield solutions
            yield (t0, t1), self.vs

            # Break if this is the last step
            if end_of_time(T, t0, t1, dt):
                break

            # If not: update members and move to next time
            self.vs_.assign(self.vs)
            t0 = t1
            t1 = t0 + dt

class BasicSingleCellSolver(BasicCardiacODESolver):
    """A basic, non-optimised solver for systems of ODEs typically
    encountered in cardiac applications of the form: find a scalar
    field :math:`v = v(t)` and a vector field :math:`s = s(t)`

    .. math::

      v_t = - I_{ion}(v, s) + I_s

      s_t = F(v, s)

    where :math:`I_{ion}` and :math:`F` are given non-linear
    functions, :math:`I_s`is some prescribed stimulus. If :math:`I_s`
    depends on time, it is assumed that :math:`I_s` is a
    :py:class:`dolfin.Expression` with parameter 't'.

    Use this solver if you just want to test the results from a
    cardiac cell model without any spatial domain dependence.

    Here, this nonlinear ODE system is solved via a theta-scheme.  By
    default theta=0.5, which corresponds to a Crank-Nicolson
    scheme. This can be changed by modifying the solver parameters.

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
        A cardiac cell model
      time (:py:class:`~dolfin.Constant` or None)
        A constant holding the current time.
      params (:py:class:`dolfin.Parameters`, optional)
        Solver parameters

    """

    def __init__(self, model, time, params=None):
        "Create solver from given cell model and optional parameters."

        assert isinstance(model, CardiacCellModel), \
            "Expecting CardiacCellModel as first argument to BasicSingleCellSolver, not %r" % model
        #assert (isinstance(time, Constant) or time is None), \
        #    "Expecting time to be a Constant instance (or None), not %r" % time
        assert isinstance(params, Parameters) or params is None, \
            "Expecting params to be a Parameters instance (or None), not %r" % params

        # Store model
        self._model = model

        # Define carefully chosen dummy domain
        domain = UnitIntervalMesh(1)

        # Extract information from cardiac cell model and ship off to
        # super-class.
        BasicCardiacODESolver.__init__(self,
                                       domain,
                                       time,
                                       model.num_states(),
                                       model.F,
                                       model.I,
                                       I_s=model.stimulus,
                                       params=params)
