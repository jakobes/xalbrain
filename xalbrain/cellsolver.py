"""This module contains solvers for (subclasses of) CellModel."""


__author__ = "Marie E. Rognes (meg@simula.no), 2012--2013"


import ufl

from xalbrain.markerwisefield import rhs_with_markerwise_field

import dolfin as df
import numpy as np

from xalbrain.cellmodels import (
    CellModel,
    MultiCellModel,
)

from xalbrain.utils import (
    state_space,
    time_stepper,
    split_function,
)

from abc import (
    ABC,
    abstractmethod
)

from operator import or_
from functools import reduce

import typing as tp


import os
import logging

from operator import or_
from functools import reduce

from xalbrain.utils import import_extension_modules


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


class AbstractCellSolver(ABC):
    """Abstract base class for cell solvers."""

    def __init__(
        self,
        *,
        mesh: df.mesh,
        time: df.Constant,
        cell_model: CellModel,
        parameters: df.Parameters = None
    ):
        """Store common parameters for all cell solvers."""
        # Initialize and update parameters if given
        self._parameters = self.default_parameters()
        if parameters is not None:
            self._parameters.update(parameters)

        # Store input
        self._mesh = mesh
        self._cell_model = cell_model

        # Create time if not given, otherwise use given time
        if time is None:
            self._time = df.Constant(0.0)
        else:
            self._time = time

        # Extract some information from cell model
        self._F = self._cell_model.F
        self._I_ion = self._cell_model.I
        self._num_states = self._cell_model.num_states()

        # Create (vector) function space for potential + states
        self.VS = df.VectorFunctionSpace(self._mesh, "CG", 1, dim=self._num_states + 1)

        # Initialize solution fields
        self.vs_ = df.Function(self.VS, name="vs_")
        self.vs = df.Function(self.VS, name="vs")

    @staticmethod
    @abstractmethod
    def default_parameters():
        """Default parameters."""
        pass

    @property
    def time(self) -> df.Constant:
        """The internal time of the solver."""
        return self._time

    def solution_fields(self) -> tp.Tuple[df.Function, df.Function]:
        """Return current solution object.

        Modifying this will modify the solution object of the solver
        and thus provides a way for setting initial conditions for
        instance.

        *Returns*
          (previous vs_, current vs) (:py:class:`dolfin.Function`)
        """
        return self.vs_, self.vs

    @abstractmethod
    def step(t0: float, t1: float) -> None:
        pass

    def solve(
        self,
        t0: float,
        t1: float,
        dt: float
    ) -> tp.Iterable[tp.Tuple[tp.Tuple[float, float], df.Function]]:
        """Solve the problem in the interval (`t0`, `t1`) with timestep `dt`.

        Arguments:
            t0: Start time.
            t1: End time.
            dt: Time step.

        Returns current solution interval and solution.

        *Example of usage*::
          # Create generator
          solutions = solver.solve((0.0, 1.0), 0.1)

          # Iterate over generator (computes solutions as you go)
          for (interval, vs) in solutions:
            # do something with the solutions

        """
        # Solve on entire interval if no interval is given.
        if dt is None:
            dt = t1 - t0

        # Create timestepper
        for _t0, _t1 in time_stepper(t0, t1, dt):
            self.step(_t0, _t1)

            # Yield solutions
            yield (_t0, _t1), self.vs
            self.vs_.assign(self.vs)


class BasicCardiacODESolver(AbstractCellSolver):
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

      model (:py:class:`xalbrain.CellModel`)
        A representation of the cardiac cell model(s)

      I_s (optional) A typically time-dependent external stimulus
        given as a :py:class:`dolfin.GenericFunction` or a
        Markerwise. NB: it is assumed that the time dependence of I_s
        is encoded via the 'time' Constant.

      parameters (:py:class:`dolfin.Parameters`, optional)
        Solver parameters
    """
    def __init__(
        self,
        mesh: df.Mesh,
        time: df.Constant,
        cell_model: CellModel,
        I_s: tp.Union[df.Expression, tp.Dict[int, df.Expression]],
        parameters: df.Parameters = None,
    ) -> None:
        """Create the necessary function spaces """
        # Handle stimulus
        self._I_s = I_s
        super().__init__(mesh=mesh, time=time, cell_model=cell_model, parameters=parameters)

    @staticmethod
    def default_parameters() -> df.Parameters:
        """Initialize and return a set of default parameters."""
        parameters = df.Parameters("BasicCardiacODESolver")
        parameters.add("theta", 0.5)

        # Use iterative solver as default.
        parameters.add(df.NonlinearVariationalSolver.default_parameters())
        parameters["nonlinear_variational_solver"]["newton_solver"]["linear_solver"] = "gmres"
        parameters["nonlinear_variational_solver"]["newton_solver"]["preconditioner"] = "jacobi"
        return parameters

    def step(self, t0: float, t1: float) -> None:
        """Solve on the given time step (`t0`, `t1`).

        End users are recommended to use solve instead.

        Arguments:
            t0: Start time.
            t1: End time.
        """
        timer = df.Timer("ODE step")

        # Extract time mesh
        k_n = df.Constant(t1 - t0)

        # Extract previous solution(s)
        v_, s_ = split_function(self.vs_, self._num_states + 1)

        # Set-up current variables
        self.vs.assign(self.vs_)     # Start with good guess
        v, s = split_function(self.vs, self._num_states + 1)
        w, r = split_function(df.TestFunction(self.VS), self._num_states + 1)

        # Define equation based on cell model
        Dt_v = (v - v_)/k_n
        Dt_s = (s - s_)/k_n

        theta = self._parameters["theta"]

        # Set time (propagates to time-dependent variables defined via self.time)
        t = t0 + theta*(t1 - t0)
        self.time.assign(t)

        v_mid = theta*v + (1.0 - theta)*v_
        s_mid = theta*s + (1.0 - theta)*s_

        if isinstance(self._cell_model, MultiCellModel):
            model = self._cell_model
            mesh = model.mesh()
            dy = df.Measure("dx", domain=mesh, subdomain_data=model.markers())

            if self._I_s is None:
                self._I_s = df.Constant(0)
            rhs = self._I_s*w*dy()

            n = model.num_states()      # Extract number of global states

            # Collect contributions to lhs by iterating over the different cell models
            domains = self._cell_model.keys()
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
        solver_parameters = self._parameters["nonlinear_variational_solver"]
        solver_parameters["nonlinear_solver"] = "snes"
        solver_parameters["snes_solver"]["absolute_tolerance"] = 1e-13
        solver_parameters["snes_solver"]["relative_tolerance"] = 1e-13

        # Tested on Cressman
        solver_parameters["snes_solver"]["linear_solver"] = "bicgstab"
        solver_parameters["snes_solver"]["preconditioner"] = "jacobi"

        solver.parameters.update(solver_parameters)

        solver.solve()
        timer.stop()


class CardiacODESolver(AbstractCellSolver):
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

      model (:py:class:`xalbrain.CellModel`)
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
            model: CellModel,
            I_s: tp.Union[df.Expression, tp.Dict[int, df.Expression]] = None,
            parameters: df.Parameters = None
    ) -> None:
        """Initialise parameters."""
        super().__init__(mesh=mesh, time=time, cell_model=model, parameters=parameters)

        import ufl.classes      # TODO Why?
        self._I_s = I_s

        # Initialize scheme
        v, s = split_function(self.vs, self._num_states + 1)
        w, q = split_function(df.TestFunction(self.VS), self._num_states + 1)

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

        name = self._parameters["scheme"]
        Scheme = self._name_to_scheme(name)
        self._scheme = Scheme(self._rhs, self.vs, self._time)

        # Initialize solver and update its parameters
        self._pi_solver = df.PointIntegralSolver(self._scheme)

    def _name_to_scheme(self, name: str) -> "df.MultiStageScheme":
        """Use the magic `eval` function to convert string to multi stage ode scheme."""
        return eval("df.multistage.{:s}".format(name))

    @staticmethod
    def default_parameters() -> df.Parameters:
        """Initialize and return a set of default parameters."""
        parameters = df.Parameters("CardiacODESolver")
        parameters.add("scheme", "RK4")
        return parameters

    def step(self, t0: float, t1: float) -> None:
        """Solve on the given time step (t0, t1).

        End users are recommended to use solve instead.

        Arguments:
            t0: Start time.
            t1: End time.
        """
        # NB: The point integral solver operates on vs directly, map initial condition in vs_ to vs:

        timer = df.Timer("ODE step")
        self.vs.assign(self.vs_)

        dt = t1 - t0
        self._pi_solver.step(dt)
        timer.stop()


class MultiCellSolver(AbstractCellSolver):
    def __init__(
        self,
        time: df.Constant,
        mesh: df.Mesh,
        cell_model: CellModel,
        parameter_map: "ODEMap",
        indicator_function: df.Function,
        parameters: df.parameters = None,
    ) -> None:
        """Initialise parameters. NB! Keep I_s for compatibility."""
        super().__init__(mesh=mesh, time=time, cell_model=cell_model, parameters=parameters)

        comm = df.MPI.comm_world
        rank = df.MPI.rank(comm)

        indicator_tags = set(np.unique(indicator_function.vector().get_local()))
        indicator_tags = comm.gather(indicator_tags, root=0)

        if rank == 0:
            indicator_tags = reduce(or_, indicator_tags)
        else:
            assert indicator_tags is None

        indicator_tags = df.MPI.comm_world.bcast(indicator_tags, root=0)
        ode_tags = set(parameter_map.get_tags())
        assert ode_tags <= indicator_tags, "Parameter map tags does not match indicator_function"
        self._indicator_function = indicator_function

        from extension_modules import load_module

        self.ode_module = load_module(
            "LatticeODESolver",
            recompile=self._parameters["reload_extension_modules"],
            verbose=self._parameters["reload_extension_modules"]
        )

        self.ode_solver = self.ode_module.LatticeODESolver(
            parameter_map,
            self.vs_.function_space().num_sub_spaces()
        )

    @staticmethod
    def default_parameters():
        parameters = df.Parameters("MultiCellSolver")
        parameters.add("reload_extension_modules", False)
        parameters.add("theta", 0.5)
        return parameters

    def step(self, t0: float, t1: float) -> None:
        """Take a step using my much better ode solver."""
        theta = self._parameters["theta"]
        dt = t1 - t0        # TODO: Is this risky?

        # Set time (propagates to time-dependent variables defined via self.time)
        t = t0 + theta*(t1 - t0)
        self._time.assign(t)

        comm = df.MPI.comm_world
        rank = df.MPI.rank(comm)

        self._indicator_function.vector()[:] = np.rint(self._indicator_function.vector().get_local())
        # assert False, np.unique(self._indicator_function.vector().get_local())
        logger.debug("MultiCell ode solver step")
        self.ode_solver.solve(self.vs_.vector(), t0, t1, dt, self._indicator_function.vector())

        logger.debug("Copy vector back")
        self.vs.vector()[:] = self.vs_.vector()[:]      # TODO: get local?
        df.MPI.barrier(comm)
        # self.vs.assign(self.vs_)


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
      model (:py:class:`~xalbrain.cellmodels.cardiaccellmodel.CellModel`)
        A cardiac cell model
      time (:py:class:`~dolfin.Constant` or None)
        A constant holding the current time.
      parameters (:py:class:`dolfin.Parameters`, optional)
        Solver parameters

    """

    def __init__(
        self,
        *,
        time: df.Constant,
        cell_model: CellModel,
        parameters: df.Parameters = None
    ) -> None:
        """Create solver from given cell model and optional parameters."""
        msg = "Expecting model to be a CellModel, not {}".format(cell_model)
        assert isinstance(cell_model, CellModel), msg

        msg = "Expecting time to be a Constant instance, not %r".format(time)
        assert (isinstance(time, df.Constant)), msg

        msg = "Expecting parameters to be a Parameters (or None), not {}".format(parameters)
        assert isinstance(parameters, df.Parameters) or parameters is None, msg

        # Define carefully chosen dummy mesh
        mesh = df.UnitIntervalMesh(1)

        super().__init__(mesh=mesh, time=time, model=cell_model, I_s=cell_model.stimulus, parameters=parameters)


class SingleCellSolver(CardiacODESolver):
    def __init__(
        self,
        *,
        cell_model: CellModel,
        time: df.Constant,
        parameters: df.Parameters=None
    ) -> None:
        """Create solver from given cell model and optional parameters."""
        assert isinstance(cell_model, CellModel), \
            "Expecting model to be a CellModel, not %r" % cell_model
        assert (isinstance(time, df.Constant)), \
            "Expecting time to be a Constant instance, not %r" % time
        assert isinstance(parameters, df.Parameters) or parameters is None, \
            "Expecting parameters to be a Parameters (or None), not %r" % parameters

        # Define carefully chosen dummy mesh
        mesh = df.UnitIntervalMesh(1)

        super().__init__(
            mesh=mesh,
            time=time,
            model=cell_model,
            I_s=cell_model.stimulus,
            parameters=parameters
        )


class SingleMultiCellSolver(MultiCellSolver):
    def __init__(
        self,
        *,
        time: df.Constant,
        cell_model: CellModel,
        parameters: df.Parameters = None
    ) -> None:
        """Create solver from given cell model and optional parameters."""
        # Define carefully chosen dummy mesh
        mesh = df.UnitIntervalMesh(1)

        _function_space = df.FunctionSpace(mesh, "CG", 1)
        indicator_function = df.Function(_function_space)
        indicator_function.vector()[:] = 1

        extension_modules = import_extension_modules()
        from extension_modules import load_module

        ode_module = load_module(
            "LatticeODESolver",
            recompile=parameters["reload_extension_modules"],
            verbose=parameters["reload_extension_modules"]
        )

        odemap = ode_module.ODEMap()
        odemap.add_ode(1, ode_module.SimpleODE())

        super().__init__(
            time=time,
            mesh=mesh,
            cell_model=cell_model,
            parameter_map=odemap,
            indicator_function=indicator_function,
            parameters=parameters
        )
