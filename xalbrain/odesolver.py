import dolfin as df

import ufl

from extension_modules import load_module

from typing import (
    Tuple,
    Dict,
    NamedTuple,
    Sequence,
    Iterator,
    Union,
)

from xalbrain.utils import (
    state_space,
    split_subspaces,
    time_stepper,
)

from xalbrain.cellmodels import (
    CellModel,
    MultiCellModel,
)


class ODESolverParameters(NamedTuple):
    parameter_map: "ODEMap"
    valid_cell_tags: Sequence[int]
    timestep: df.Constant = df.Constant(1)
    reload_extension_modules: bool = False
    theta: df.Constant = df.Constant(0.5)


class ODESolver:
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

      model (:py:class:`xalbrain.cCellModel`)
        A representation of the cardiac cell model(s)

      I_s (:py:class:`dolfin.Expression`, optional)
        A typically time-dependent external stimulus. NB: it is
        assumed that the time dependence of I_s is encoded via the
        'time' Constant.

      parameters (:py:class:`dolfin.Parameters`, optional)
        Solver parameters
    """

    def __init__(
            self,
            mesh: df.Mesh,
            time: df.Constant,
            cell_model: CellModel,
            I_s: Union[df.Expression, Dict[int, df.Expression]]=None,
            parameters: df.Parameters=None
    ) -> None:
        """Initialise parameters."""
        import ufl.classes

        # Store input
        self._mesh = mesh
        self._time = time
        self._cell_model = cell_model

        # Extract some information from cell model
        self._F = self._cell_model.F
        self._I_ion = self._cell_model.I
        self._num_states = self._cell_model.num_states()

        if I_s is None:
            self._I_s = df.Constant(0)
        else:
            self._I_s = I_s

        # Create time if not given, otherwise use given time
        if time is None:
            self._time = df.Constant(0.0)
        else:
            self._time = time

        # Initialize and update parameters if given
        self.parameters = self.default_parameters()
        if parameters is not None:
            self.parameters.update(parameters)

        # Create (vector) function space for potential + states
        self.VS = df.VectorFunctionSpace(
            self._mesh,
            "CG",
            1,
            dim=self._num_states + 1
        )

        # Initialize solution field
        self.vs_ = df.Function(self.VS, name="vs_")
        self.vs = df.Function(self.VS, name="vs")

        # Initialize scheme
        v, s = split_subspaces(self.vs, self._num_states + 1)
        w, q = split_subspaces(df.TestFunction(self.VS), self._num_states + 1)

        # Workaround to get algorithm in RL schemes working as it only
        # works for scalar expressions
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

    def _name_to_scheme(self, name):
        """Return scheme class with given name.

        *Arguments*
          name (string)

        *Returns*
          the Scheme (:py:class:`dolfin.MultiStageScheme`)

        """
        return eval("df.{:s}".format(name))

    @staticmethod
    def default_parameters() -> df.Parameters:
        """Initialize and return a set of default parameters

        *Returns*
          A set of parameters (:py:class:`dolfin.Parameters`)
        """
        parameters = df.Parameters("ODESolver")
        parameters.add("scheme", "RK4")
        # parameters.add(df.PointIntegralSolver.default_parameters())
        # parameters.add("enable_adjoint", False)

        return parameters

    @property
    def time(self) -> df.Constant:
        """The internal time of the solver."""
        return self._time

    def solution_fields(self) -> Tuple[df.Function, df.Function]:
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
        # NB: The point integral solver operates on vs directly, map
        # initial condition in vs_ to vs:

        timer = df.Timer("ODE step")
        self.vs.assign(self.vs_)

        dt = t1 - t0
        self._pi_solver.step(dt)
        timer.stop()

    def solve(self, interval: Tuple[float, float], dt: float=None) -> None:
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
        T0, T = interval

        # Solve on entire interval if no interval is given.
        for t0, t1 in time_stepper(t0=t0, t1=t1, dt=dt):
            # df.info_blue("Solving on t = (%g, %g)" % (t0, t1))
            self.step(t0, t1)

            # Yield solutions
            yield (t0, t1), self.vs

            # FIXME: This eventually breaks in parallel!?
            self.vs_.assign(self.vs)


class SubDomainODESolver:
    def __init__(
            self,
            time: df.Constant,
            mesh: df.Mesh,
            cell_model: CellModel,
            parameters: ODESolverParameters,
            cell_function: df.MeshFunction,
    ) -> None:
        """Initialise parameters. NB! Keep I_s for compatibility"""
        # Store input
        self._mesh = mesh
        self._time = time
        self._model = cell_model     # FIXME: For initial conditions and num states

        # Extract some information from cell model
        self._num_states = self._model.num_states()

        self._parameters = parameters

        if cell_function is None:
            cell_function = df.MeshFunction("size_t", mesh, mesh.geometric_dimension())

        _cell_function_tags = set(cell_function.array())
        if not set(self._parameters.valid_cell_tags) <= _cell_function_tags:
            msg = "Valid cell tag not found in cell function. Expected {}, for {}."
            raise ValueError(msg.format(set(self._parameters.valid_cell_tags), _cell_function_tags))
        valid_cell_tags = self._parameters.valid_cell_tags

        # Create (vector) function space for potential + states
        self._function_space_VS = df.VectorFunctionSpace(self._mesh, "CG", 1, dim=self._num_states + 1)

        # Initialize solution field
        self.vs_prev = df.Function(self._function_space_VS, name="vs_prev")
        self.vs = df.Function(self._function_space_VS, name="vs")

        model_name = cell_model.__class__.__name__        # Which module to load
        self.ode_module = load_module(
            "LatticeODESolver",
            recompile=self._parameters.reload_extension_modules,
            verbose=self._parameters.reload_extension_modules
        )

        self.ode_solver = self.ode_module.LatticeODESolver(
            self._function_space_VS._cpp_object,
            cell_function,
            self._parameters.parameter_map,
        )

    def solution_fields(self) -> Tuple[df.Function, df.Function]:
        """
        Return current solution object.

        Modifying this will modify the solution object of the solver
        and thus provides a way for setting initial conditions for
        instance.
        """
        return self.vs_prev, self.vs

    def step(self, t0: float, t1: float) -> None:
        """Take a step using my much better ode solver."""
        theta = self._parameters.theta
        dt = t1 - t0        # TODO: Is this risky?

        # Set time (propagates to time-dependent variables defined via self.time)
        t = t0 + theta*(t1 - t0)
        self._time.assign(t)

        # FIXME: Is there some theta shenanigans I have missed?
        self.ode_solver.solve(self.vs_prev.vector(), t0, t1, dt)
        self.vs.assign(self.vs_prev)

    def solve(
            self,
            t0: float,
            t1: float,
            dt: float = None,
    ) -> Iterator[Tuple[Tuple[float, float], df.Function]]:
        """
        Solve the problem given by the model on a given time interval
        (t0, t1) with a given timestep dt and return generator for a
        tuple of the interval and the current vs solution.

        *Example of usage*::

          # Create generator
          solutions = solver.solve(0.0, 1.0, 0.1)

          # Iterate over generator (computes solutions as you go)
          for interval, vs in solutions:
            # do something with the solutions

        """
        # Solve on entire interval if no interval is given.
        for interval in time_stepper(t0=t0, t1=t1, dt=dt):
            self.step(*interval)

            # Yield solutions
            yield interval, self.vs
            self.vs_prev.assign(self.vs)


class SubDomainSingleCellSolver(SubDomainODESolver):
    def __init__(
            self,
            cell_model: CellModel,
            time: df.Constant,
            reload_ext_modules: bool = False,
            parameters: df.Parameters = None
    ) -> None:
        """Create solver from given cell model and optional parameters."""
        # Store model
        self.cell_model = cell_model

        # Define carefully chosen dummy mesh
        mesh = df.UnitIntervalMesh(1)

        super().__init__(
            mesh,
            time,
            cell_model,
            reload_ext_modules=reload_ext_modules,
            parameters=parameters
        )


class SingleCellSolver(ODESolver):
    def __init__(
            self,
            cell_model: CellModel,
            time: df.Constant,
            parameters: df.Parameters=None
    ) -> None:
        """Create solver from given cell model and optional parameters."""
        # Store model
        self._cell_model = cell_model

        # Define carefully chosen dummy mesh
        mesh = df.UnitIntervalMesh(1)

        # Extract information from cardiac cell model and ship off to
        # super-class.
        ODESolver.__init__(
            self,
            mesh,
            time,
            cell_model,
            I_s=cell_model.stimulus,
            parameters=parameters
        )


