"This module contains solvers for (subclasses of) CardiacCellModel."


__author__ = "Marie E. Rognes (meg@simula.no), 2012--2013"


import ufl

from xalbrain.markerwisefield import rhs_with_markerwise_field

import dolfin as df

from xalbrain.cellmodels import (
    CardiacCellModel,
    MultiCellModel,
)

from xalbrain.utils import (
    state_space,
    TimeStepper,
    splat,
)

import typing as tp


class BasicCardiacODESolver:
    """A basic, non-optimised solver for systems of ODEs typically
    encountered in cardiac applications of the form: find a scalar
    field :math:`v = v(x, t)` and a vector field :math:`s = s(x, t)`

    .. math::

      v_t = - I_{ion}(v, s) + I_s

      s_t = F(v, s)

    where :math:`I_{ion}` and :math:`F` are given non-linear
    functions, and :math:`I_s` is some prescribed stimulus.

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
      mesh (:py:class:`dolfin.Mesh`)
        The spatial domain (mesh)

      time (:py:class:`dolfin.Constant` or None)
        A constant holding the current time. If None is given, time is
        created for you, initialized to zero.

      model (:py:class:`xalbrain.CardiacCellModel`)
        A representation of the cardiac cell model(s)

      I_s (optional) A typically time-dependent external stimulus
        given as a :py:class:`dolfin.GenericFunction` or a
        Markerwise. NB: it is assumed that the time dependence of I_s
        is encoded via the 'time' Constant.

      params (:py:class:`dolfin.Parameters`, optional)
        Solver parameters
    """
    def __init__(
            self,
            mesh: df.Mesh,
            time: df.Constant,
            model: CardiacCellModel,
            I_s: tp.Union[df.Expression, tp.Dict[int, df.Expression]],
            params: df.Parameters = None,
    ) -> None:
        """Create the necessary function spaces """
        # Store input
        self._mesh = mesh
        self._time = time
        self._model = model

        # Extract some information from cell model
        self._F = self._model.F
        self._I_ion = self._model.I
        self._num_states = self._model.num_states()

        # Handle stimulus
        self._I_s = I_s

        # Initialize and update parameters if given
        self.parameters = self.default_parameters()
        if params is not None:
            self.parameters.update(params)

        # Create (mixed) function space for potential + states
        v_family = self.parameters["V_polynomial_family"]
        v_degree = self.parameters["V_polynomial_degree"]
        s_family = self.parameters["S_polynomial_family"]
        s_degree = self.parameters["S_polynomial_degree"]

        if v_family == s_family and s_degree == v_degree:
            self.VS = df.VectorFunctionSpace(self._mesh, v_family, v_degree, dim=self._num_states + 1)
        else:
            V = df.FunctionSpace(self._mesh, v_family, v_degree)
            S = state_space(self._mesh, self._num_states, s_family, s_degree)
            Mx = df.MixedElement(V.ufl_element(), S.ufl_element())
            self.VS = df.FunctionSpace(self._mesh, Mx)

        # Initialize solution fields
        self.vs_ = df.Function(self.VS, name="vs_")
        self.vs = df.Function(self.VS, name="vs")

    @property
    def time(self) -> df.Constant:
        "The internal time of the solver."
        return self._time

    @staticmethod
    def default_parameters() -> df.Parameters:
        """Initialize and return a set of default parameters

        *Returns*
          A set of parameters (:py:class:`dolfin.Parameters`)
        """
        params = df.Parameters("BasicCardiacODESolver")
        params.add("theta", 0.5)
        params.add("V_polynomial_degree", 0)
        params.add("V_polynomial_family", "DG")
        params.add("S_polynomial_degree", 0)
        params.add("S_polynomial_family", "DG")
        params.add("enable_adjoint", False)

        # Use iterative solver as default.
        params.add(df.NonlinearVariationalSolver.default_parameters())
        params["nonlinear_variational_solver"]["newton_solver"]["linear_solver"] = "gmres"
        params["nonlinear_variational_solver"]["newton_solver"]["preconditioner"] = "jacobi"
        return params

    def solution_fields(self) -> tp.Tuple[df.Function, df.Function]:
        """
        Return tuple of previous and current solution objects.

        Modifying these will modify the solution objects of the solver
        and thus provides a way for setting initial conditions for
        instance.

        *Returns*
          (previous vs, current vs) (:py:class:`tuple` of :py:class:`dolfin.Function`)
        """
        return self.vs_, self.vs

    def solve(self, t0: float, t1: float, dt: float) -> tp.Any:
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
        # Solve on entire interval if no interval is given.
        if dt is None:
            dt = t1 - t0

        # Create timestepper
        time_stepper = TimeStepper(t0, t1, dt)
        for _t0, _t1 in time_stepper:

            # df.info_blue("Solving on t = ({:g}, {:g})".format(t0, t1))
            self.step(_t0, _t1)

            # Yield solutions
            yield (_t0, _t1), self.vs
            self.vs_.assign(self.vs)

    def step(self, t0: float, t1: float) -> None:
        """
        Solve on the given time step (t0, t1).

        End users are recommended to use solve instead.

        *Arguments*
          interval (:py:class:`tuple`)
            The time interval (t0, t1) for the step
        """
        timer = df.Timer("ODE step")

        # Extract time mesh
        k_n = df.Constant(t1 - t0)

        # Extract previous solution(s)
        v_, s_ = splat(self.vs_, self._num_states + 1)

        # Set-up current variables
        self.vs.assign(self.vs_)     # Start with good guess
        v, s = splat(self.vs, self._num_states + 1)
        w, r = splat(df.TestFunction(self.VS), self._num_states + 1)

        # Define equation based on cell model
        Dt_v = (v - v_)/k_n
        Dt_s = (s - s_)/k_n

        theta = self.parameters["theta"]

        # Set time (propagates to time-dependent variables defined via self.time)
        t = t0 + theta*(t1 - t0)
        self.time.assign(t)

        v_mid = theta*v + (1.0 - theta)*v_
        s_mid = theta*s + (1.0 - theta)*s_

        if isinstance(self._model, MultiCellModel):
            model = self._model
            mesh = model.mesh()
            dy = df.Measure("dx", domain=mesh, subdomain_data=model.markers())

            if self._I_s is None:
                self._I_s = df.Constant(0)
            rhs = self._I_s*w*dy()

            n = model.num_states()      # Extract number of global states

            # Collect contributions to lhs by iterating over the different cell models
            domains = self._model.keys()
            lhs_list = list()
            for k, model_k in enumerate(model.models()):
                n_k = model_k.num_states()      # Extract number of local (non-trivial) states

                # Extract right components of coefficients and test functions () is not the same as (1,)
                if n_k == 1:
                    s_mid_k = s_mid[0]
                    r_k = r[0]
                    Dt_s_k = Dt_s[0]
                else:
                    s_mid_k = df.as_vector(tuple(s_mid[j] for j in range(n_k)))
                    r_k = df.as_vector(tuple(r[j] for j in range(n_k)))
                    Dt_s_k = df.as_vector(tuple(Dt_s[j] for j in range(n_k)))

                i_k = domains[k]        # Extract domain index of cell model k

                # Extract right currents and ion channel expressions
                F_theta_k = self._F(v_mid, s_mid_k, time=self.time, index=i_k)
                I_theta_k = -self._I_ion(v_mid, s_mid_k, time=self.time, index=i_k)

                # Variational contribution over the relevant domain
                a_k = (
                    (Dt_v - I_theta_k)*w
                    + df.inner(Dt_s_k, r_k)
                    - df.inner(F_theta_k, r_k)
                )*dy(i_k)

                # Add s_trivial = 0 on Omega_{i_k} in variational form:
                a_k += sum(s[j]*r[j] for j in range(n_k, n))*dy(i_k)
                lhs_list.append(a_k)
            lhs = sum(lhs_list)
        else:
            dz, rhs = rhs_with_markerwise_field(self._I_s, self._mesh, w)

            # Evaluate currents at averaged v and s. Note sign for I_theta
            F_theta = self._F(v_mid, s_mid, time=self.time)
            I_theta = -self._I_ion(v_mid, s_mid, time=self.time)
            lhs = (Dt_v - I_theta)*w*dz + df.inner(Dt_s - F_theta, r)*dz

        # Set-up system of equations
        G = lhs - rhs

        # Solve system
        pde = df.NonlinearVariationalProblem(G, self.vs, J=df.derivative(G, self.vs))
        solver = df.NonlinearVariationalSolver(pde)
        solver_params = self.parameters["nonlinear_variational_solver"]
        solver_params["nonlinear_solver"] = "snes"
        solver_params["snes_solver"]["absolute_tolerance"] = 1e-13
        solver_params["snes_solver"]["relative_tolerance"] = 1e-13

        # Tested on Cressman
        solver_params["snes_solver"]["linear_solver"] = "bicgstab"
        solver_params["snes_solver"]["preconditioner"] = "jacobi"

        solver.parameters.update(solver_params)

        solver.solve()
        timer.stop()


class CardiacODESolver:
    """An optimised solver for systems of ODEs typically
    encountered in cardiac applications of the form: find a scalar
    field :math:`v = v(x, t)` and a vector field :math:`s = s(x, t)`

    .. math::

      v_t = - I_{ion}(v, s) + I_s

      s_t = F(v, s)

    where :math:`I_{ion}` and :math:`F` are given non-linear
    functions, and :math:`I_s` is some prescribed stimulus.

    .. note::

       For the sake of simplicity and consistency with other solver
       objects, this solver operates on its solution fields (as state
       variables) directly internally. More precisely, solve (and
       step) calls will act by updating the internal solution
       fields. It implies that initial conditions can be set (and are
       intended to be set) by modifying the solution fields prior to
       simulation.

    *Arguments*
      mesh (:py:class:`dolfin.Mesh`)
        The spatial mesh (mesh)

      time (:py:class:`dolfin.Constant` or None)
        A constant holding the current time. If None is given, time is
        created for you, initialized to zero.

      model (:py:class:`xalbrain.CardiacCellModel`)
        A representation of the cardiac cell model(s)

      I_s (:py:class:`dolfin.Expression`, optional)
        A typically time-dependent external stimulus. NB: it is
        assumed that the time dependence of I_s is encoded via the
        'time' Constant.

      params (:py:class:`dolfin.Parameters`, optional)
        Solver parameters
    """

    def __init__(
            self,
            mesh: df.Mesh,
            time: df.Constant,
            model: CardiacCellModel,
            I_s: tp.Union[df.Expression, tp.Dict[int, df.Expression]]=None,
            params: df.Parameters=None
    ) -> None:
        """Initialise parameters."""
        import ufl.classes

        # Store input
        self._mesh = mesh
        self._time = time
        self._model = model

        # Extract some information from cell model
        self._F = self._model.F
        self._I_ion = self._model.I
        self._num_states = self._model.num_states()

        self._I_s = I_s

        # Create time if not given, otherwise use given time
        if time is None:
            self._time = df.Constant(0.0)
        else:
            self._time = time

        # Initialize and update parameters if given
        self.parameters = self.default_parameters()
        if params is not None:
            self.parameters.update(params)

        # Create (vector) function space for potential + states
        self.VS = df.VectorFunctionSpace(self._mesh, "CG", 1, dim=self._num_states + 1)

        # Initialize solution field
        self.vs_ = df.Function(self.VS, name="vs_")
        self.vs = df.Function(self.VS, name="vs")

        # Initialize scheme
        v, s = splat(self.vs, self._num_states + 1)
        w, q = splat(df.TestFunction(self.VS), self._num_states + 1)

        # Workaround to get algorithm in RL schemes working as it only works for scalar expressions
        F_exprs = self._F(v, s, self._time)

        # MER: This looks much more complicated than it needs to be!
        # If we have a as_vector expression
        F_exprs_q = ufl.zero()
        if isinstance(F_exprs, ufl.classes.ListTensor):
            for i, expr_i in enumerate(F_exprs.ufl_operands):
                F_exprs_q += expr_i*q[i]
        else:
            F_exprs_q = F_exprs*q

        rhs = F_exprs_q - self._I_ion(v, s, self._time)*w

        # Handle stimulus: only handle single function case for now
        if self._I_s:
            rhs += self._I_s*w

        self._rhs = rhs*df.dP()

        name = self.parameters["scheme"]
        Scheme = self._name_to_scheme(name)
        self._scheme = Scheme(self._rhs, self.vs, self._time)

        # Initialize solver and update its parameters
        self._pi_solver = df.PointIntegralSolver(self._scheme)

    def _name_to_scheme(self, name: str) -> tp.Any:
        """Return scheme class with given name.

        *Arguments*
          name (string)

        *Returns*
          the Scheme (:py:class:`dolfin.MultiStageScheme`)

        """
        return eval("df.multistage.{:s}".format(name))

    @staticmethod
    def default_parameters() -> df.Parameters:
        """Initialize and return a set of default parameters

        *Returns*
          A set of parameters (:py:class:`dolfin.Parameters`)
        """
        params = df.Parameters("CardiacODESolver")
        params.add("scheme", "RK4")
        return params

    @property
    def time(self) -> df.Constant:
        """The internal time of the solver."""
        return self._time

    def solution_fields(self) -> tp.Tuple[df.Function, df.Function]:
        """
        Return current solution object.

        Modifying this will modify the solution object of the solver
        and thus provides a way for setting initial conditions for
        instance.

        *Returns*
          (previous vs_, current vs) (:py:class:`dolfin.Function`)
        """
        return self.vs_, self.vs

    def step(self, t0: float, t1: float) -> None:
        """
        Solve on the given time step (t0, t1).

        End users are recommended to use solve instead.

        *Arguments*
          interval (:py:class:`tuple`)
            The time interval (t0, t1) for the step
        """
        # NB: The point integral solver operates on vs directly, map initial condition in vs_ to vs:

        timer = df.Timer("ODE step")
        self.vs.assign(self.vs_)

        dt = t1 - t0
        self._pi_solver.step(dt)
        timer.stop()

    def solve(self, t0: float, t1: float, dt: float) -> None:
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
        # Solve on entire interval if no interval is given.
        if dt is None:
            dt = t1 - t0

        # Create timestepper
        time_stepper = TimeStepper(t0, t1, dt)

        for _t0, _t1 in time_stepper:
            # df.info_blue("Solving on t = (%g, %g)" % (t0, t1))
            self.step(_t0, _t1)

            # Yield solutions
            yield (_t0, _t1), self.vs
            self.vs_.assign(self.vs)


class BasicSingleCellSolver(BasicCardiacODESolver):
    """A basic, non-optimised solver for systems of ODEs typically
    encountered in cardiac applications of the form: find a scalar
    field :math:`v = v(t)` and a vector field :math:`s = s(t)`

    .. math::

      v_t = - I_{ion}(v, s) + I_s

      s_t = F(v, s)

    where :math:`I_{ion}` and :math:`F` are given non-linear
    functions, :math:`I_s` is some prescribed stimulus. If :math:`I_s`
    depends on time, it is assumed that :math:`I_s` is a
    :py:class:`dolfin.Expression` with parameter 't'.

    Use this solver if you just want to test the results from a
    cardiac cell model without any spatial mesh dependence.

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
      model (:py:class:`~xalbrain.cellmodels.cardiaccellmodel.CardiacCellModel`)
        A cardiac cell model
      time (:py:class:`~dolfin.Constant` or None)
        A constant holding the current time.
      params (:py:class:`dolfin.Parameters`, optional)
        Solver parameters

    """

    def __init__(
            self,
            model: CardiacCellModel,
            time: df.Constant,
            params: df.Parameters = None
    ) -> None:
        """Create solver from given cell model and optional parameters."""
        msg = "Expecting model to be a CardiacCellModel, not {}".format(model)
        assert isinstance(model, CardiacCellModel), msg

        msg = "Expecting time to be a Constant instance, not %r".format(time)
        assert (isinstance(time, df.Constant)), msg

        msg = "Expecting params to be a Parameters (or None), not {}".format(params)
        assert isinstance(params, df.Parameters) or params is None, msg

        # Store model
        self._model = model

        # Define carefully chosen dummy mesh
        mesh = df.UnitIntervalMesh(1)

        super().__init__(mesh, time, model, I_s=model.stimulus, params=params)


class SingleCellSolver(CardiacODESolver):
    def __init__(
            self,
            model: CardiacCellModel,
            time: df.Constant,
            params: df.Parameters=None
    ) -> None:
        """Create solver from given cell model and optional parameters."""
        assert isinstance(model, CardiacCellModel), \
            "Expecting model to be a CardiacCellModel, not %r" % model
        assert (isinstance(time, df.Constant)), \
            "Expecting time to be a Constant instance, not %r" % time
        assert isinstance(params, df.Parameters) or params is None, \
            "Expecting params to be a Parameters (or None), not %r" % params

        # Store model
        self._model = model

        # Define carefully chosen dummy mesh
        mesh = df.UnitIntervalMesh(1)

        super().__init__(mesh, time, model, I_s=model.stimulus, params=params)
