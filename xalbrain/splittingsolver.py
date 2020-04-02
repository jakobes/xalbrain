"""This module contains splitting solvers for Model objects. 
In particular, the classes

  * SplittingSolver
  * BasicSplittingSolver

These solvers solve the bidomain (or monodomain) equations on the
form: find the transmembrane potential :math:`v = v(x, t)` in mV, the
extracellular potential :math:`u = u(x, t)` in mV, and any additional
state variables :math:`s = s(x, t)` such that

.. math::

   v_t - \mathrm{div} (M_i \mathrm{grad} v + M_i \mathrm{grad} u) = - I_{ion}(v, s) + I_s

         \mathrm{div} (M_i \mathrm{grad} v + (M_i + M_e) \mathrm{grad} u) = I_a

   s_t = F(v, s)

where

  * the subscript :math:`t` denotes the time derivative,
  * :math:`M_i` and :math:`M_e` are conductivity tensors (in mm^2/ms)
  * :math:`I_s` is prescribed input current (in mV/ms)
  * :math:`I_a` is prescribed input current (in mV/ms)
  * :math:`I_{ion}` and :math:`F` are typically specified by a cell model

Note that M_i and M_e can be viewed as scaled by :math:`\chi*C_m` where
  * :math:`\chi` is the surface-to volume ratio of cells (in 1/mm) ,
  * :math:`C_m` is the specific membrane capacitance (in mu F/(mm^2) ),

In addition, initial conditions are given for :math:`v` and :math:`s`:

.. math::

   v(x, 0) = v_0

   s(x, 0) = s_0

Finally, boundary conditions must be prescribed. These solvers assume
pure Neumann boundary conditions for :math:`v` and :math:`u` and
enforce the additional average value zero constraint for u.

The solvers take as input a
:py:class:`~xalbrain.cardiacmodels.Model` providing the
required input specification of the problem. In particular, the
applied current :math:`I_a` is extracted from the
:py:attr:`~xalbrain.cardiacmodels.Model.applied_current`
attribute, while the stimulus :math:`I_s` is extracted from the
:py:attr:`~xalbrain.cardiacmodels.Model.stimulus` attribute.

It should be possible to use the solvers interchangably. However, note
that the BasicSplittingSolver is not optimised and should be used for
testing or debugging purposes primarily.
"""

# Copyright (C) 2012-2013 Marie E. Rognes (meg@simula.no)
# Use and modify at will
# Last changed: 2013-04-15

__all__ = ["SplittingSolver", "BasicSplittingSolver"]

import dolfin as df
import numpy as np

from xalbrain import (
    Model,
    MultiCellModel,
)

from xalbrain.cellsolver import (
    BasicCardiacODESolver,
    CardiacODESolver,
    MultiCellSolver
)

from xalbrain.bidomainsolver import (
    BasicBidomainSolver,
    BidomainSolver,
)

from xalbrain.monodomainsolver import (
    BasicMonodomainSolver,
    MonodomainSolver,
)

from xalbrain.utils import time_stepper

from abc import (
    ABC,
    abstractmethod,
)

import typing as tp

import time

import logging
import os


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


class AbstractSplittingSolver(ABC):
    def __init__(
            self,
            model: Model,
            ode_timestep: float = None
    ) -> None:
        """Create solver from given Cardiac Model and (optional) parameters."""
        assert isinstance(model, Model), "Expecting Model as first argument"

        self._ode_timestep = ode_timestep

        # Set model and parameters
        self._model = model

        # Extract solution domain
        self._domain = self._model.mesh
        self._time = self._model.time
        self._cell_function = self._model.cell_domains

        # Create ODE solver and extract solution fields
        self.ode_solver = self._create_ode_solver()
        self.vs_, self.vs = self.ode_solver.solution_fields()
        self.VS = self.vs.function_space()

        # Create PDE solver and extract solution fields
        self.pde_solver = self._create_pde_solver()
        self.v_, self.vur = self.pde_solver.solution_fields()

        # # Create function assigner for merging v from self.vur into self.vs[0]
        if self._parameters["pde_solver"] == "bidomain":
            V = self.vur.function_space().sub(0)
        else:
            V = self.vur.function_space()
        self.merger = df.FunctionAssigner(self.VS.sub(0), V)

    @abstractmethod
    def _create_ode_solver(self):
        pass

    @abstractmethod
    def _create_pde_solver(self):
        pass

    def solution_fields(self) -> tp.Tuple[df.Function, df.Function, df.Function]:
        """Return tuple of previous and current solution objects.

        Modifying these will modify the solution objects of the solver
        and thus provides a way for setting initial conditions for
        instance.

        Returns vs_, vs, vur
        """
        return self.vs_, self.vs, self.vur

    def solve(
        self,
        t0: float,
        t1: float,
        dt: float
    ) -> tp.Iterator[tp.Tuple[tp.Tuple[float, float], df.Function]]:
        """Solve the problem given by the model on a time interval with a given time step.

        Return a generator for a tuple of the time step and the solution fields.

        Arguments;
            interval: The time interval for the solve given by (t0, t1)
            dt: The timestep for the solve.

        Returns: timestep, solution_fields

        Example of usage::

          # Create generator
          solutions = solver.solve(t0=0.0, t1=1.0, dt=0.01)

          # Iterate over generator (computes solutions as you go)
          for ((t0, t1), (vs_, vs, vur)) in solutions:
            # do something with the solutions
        """
        # Create timestepper
        for _t0, _t1 in time_stepper(t0, t1, dt):
            self.step(_t0, _t1)

            # Yield solutions
            yield (_t0, _t1), self.solution_fields()

            # Update previous solution
            self.vs_.assign(self.vs)

    def step(self, t0: float, t1: float) -> None:
        """Solve the pde for one time step.

        Arguments:
            t0: Start time
            t1: End time

        Invariants:
          Given self._vs in a correct state at t0, provide v and s (in self.vs) and u (in self.vur) in a correct state at t1. (Note
          that self.vur[0] == self.vs[0] only if theta = 1.0.)
        """
        theta = self._parameters["theta"]

        dt = t1 - t0
        t = t0 + theta*dt

        # Compute tentative membrane potential and state (vs_star) Assumes that its vs_ is in the
        # correct state, gives its vs in the current state

        self.ode_solver.step(t0, t)
        self.vs_.assign(self.vs)

        # Compute tentative potentials vu = (v, u) Assumes that its vs_ is in the correct state,
        # gives vur in the current state
        self.pde_solver.step(t0, t1)

        # If first order splitting, we need to ensure that self.vs is
        # up to date, but otherwise we are done.
        if theta == 1.0:
            # Assumes that the v part of its vur and the s part of its vs are in the correct state,
            # provides input argument(in this case self.vs) in its correct state
            self.merge(self.vs)
            return

        # Otherwise, we do another ode_step:

        # Assumes that the v part of its vur and the s part of its vs are in the correct state,
        # provides input argument (in this # case self.vs_) in its correct state
        self.merge(self.vs_)    # self.vs_.sub(0) <- self.vur.sub(0)
        # Assumes that its vs_ is in the correct state, provides vs in the correct state

        self.ode_solver.step(t, t1)

    def merge(self, solution: df.Function) -> None:
        """Combine solutions from the PDE and the ODE to form a single mixed function.

        Arguments:
            function holding the combined result
        """
        timer = df.Timer("Merge step")
        if self._parameters["pde_solver"] == "bidomain":
            v = self.vur.sub(0)
        else:
            v = self.vur
        self.merger.assign(solution.sub(0), v)
        timer.stop()

    @property
    def model(self) -> Model:
        """Return the brain."""
        return self._model

    @property
    def parameters(self) -> df.Parameters:
        """Return the parameters."""
        return self._parameters


class BasicSplittingSolver(AbstractSplittingSolver):
    """
    A non-optimised solver for the bidomain equations based on the
    operator splitting scheme described in Sundnes et al 2006, p. 78
    ff.

    The solver computes as solutions:

      * "vs" (:py:class:`dolfin.Function`) representing the solution
        for the transmembrane potential and any additional state
        variables, and
      * "vur" (:py:class:`dolfin.Function`) representing the
        transmembrane potential in combination with the extracellular
        potential and an additional Lagrange multiplier.

    The algorithm can be controlled by a number of parameters. In
    particular, the splitting algorithm can be controlled by the
    parameter "theta": "theta" set to 1.0 corresponds to a (1st order)
    Godunov splitting while "theta" set to 0.5 to a (2nd order) Strang
    splitting.

    This solver has not been optimised for computational efficiency
    and should therefore primarily be used for debugging purposes. For
    an equivalent, but more efficient, solver, see
    :py:class:`xalbrain.splittingsolver.SplittingSolver`.

    *Arguments*
      model (:py:class:`xalbrain.cardiacmodels.Model`)
        a Model object describing the simulation set-up
      parameters (:py:class:`dolfin.Parameters`, optional)
        a Parameters object controlling solver parameters

    *Assumptions*
      * The cardiac conductivities do not vary in time
    """

    def __init__(
        self,
        model: Model,
        ode_timestep: float = None,
        parameters: df.Parameters = None
    ) -> None:
        self._parameters = self.default_parameters()
        if parameters is not None:
            self._parameters.update(parameters)

        super().__init__(model, ode_timestep)

    def _create_ode_solver(self):
        """Helper function to initialize a suitable ODE solver from the cardiac model."""
        cell_model = self._model.cell_models

        # Extract stimulus from the cardiac model(!)
        if self._parameters["apply_stimulus_current_to_pde"]:
            stimulus = self._model.stimulus()
        else:
            stimulus = None

        parameters = self._parameters["BasicCardiacODESolver"]
        solver = BasicCardiacODESolver(
            self._domain,
            self._time,
            cell_model,
            I_s=stimulus,
            parameters=parameters
        )
        return solver

    def _create_pde_solver(self):
        """Helper function to initialize a suitable PDE solver from the cardiac model."""
        applied_current = self._model.applied_current()
        ect_current =  self._model.ect_current

        # Extract stimulus from the cardiac model if we should apply
        # it to the PDEs (in the other case, it is handled by the ODE solver)
        if self._parameters["apply_stimulus_current_to_pde"]:
            stimulus = None
        else:
            stimulus = self._model.stimulus()

        # Extract conductivities from the cardiac model
        Mi, Me = self._model.conductivities()

        if self._parameters["pde_solver"] == "bidomain":
            PDESolver = BasicBidomainSolver
            parameters = self._parameters["BasicBidomainSolver"]
            parameters["theta"] = self._parameters["theta"]
            pde_args = (self._domain, self._time, Mi, Me)
            pde_kwargs = dict(
                I_s = stimulus,
                I_a = applied_current,
                ect_current = ect_current,
                v_ = self.vs[0],
                cell_domains = self._model.cell_domains,
                facet_domains = self._model.facet_domains,
                dirichlet_bc = self._model.dirichlet_bc_u,          # dirichlet_bc
                dirichlet_bc_v = self._model.dirichlet_bc_v,        # dirichlet_bc
                parameters = parameters
            )
        else:
            PDESolver = BasicMonodomainSolver
            parameters = self._parameters["BasicMonodomainSolver"]
            pde_args = (self._domain, self._time, Mi)
            pde_kwargs = dict(
                I_s = stimulus,
                v_ = self.vs[0],
                parameters = parameters,
                cell_domains = self._model.cell_domains,
                facet_domains = self._model.facet_domains,
            )
        solver = PDESolver(*pde_args, **pde_kwargs)
        return solver

    @staticmethod
    def default_parameters() -> df.Parameters:
        """
        Initialize and return a set of default parameters for the splitting solver.

        *Returns*
          A set of parameters (:py:class:`dolfin.Parameters`)

        To inspect all the default parameters, do::

          info(BasicSplittingSolver.default_parameters(), True)
        """
        parameters = df.Parameters("BasicSplittingSolver")
        parameters.add("theta", 0.5, 0., 1.)
        parameters.add("apply_stimulus_current_to_pde", False)
        parameters.add("pde_solver", "bidomain")

        # Add default parameters from ODE solver, but update for V space
        ode_solver_parameters = BasicCardiacODESolver.default_parameters()
        parameters.add(ode_solver_parameters)

        pde_solver_parameters = BasicBidomainSolver.default_parameters()
        pde_solver_parameters["polynomial_degree"] = 1
        parameters.add(pde_solver_parameters)

        pde_solver_parameters = BasicMonodomainSolver.default_parameters()
        pde_solver_parameters["polynomial_degree"] = 1
        parameters.add(pde_solver_parameters)
        return parameters


class SplittingSolver(AbstractSplittingSolver):
    """
    An optimised solver for the bidomain equations.
    The solver is based on the operator splitting scheme described in 
    Sundnes et al 2006, p. 78.

    The solver computes as solutions:

      * "vs" (:py:class:`dolfin.Function`) representing the solution
        for the transmembrane potential and any additional state
        variables, and
      * "vur" (:py:class:`dolfin.Function`) representing the
        transmembrane potential in combination with the extracellular
        potential and an additional Lagrange multiplier.

    The algorithm can be controlled by a number of parameters. In
    particular, the splitting algorithm can be controlled by the
    parameter "theta": "theta" set to 1.0 corresponds to a (1st order)
    Godunov splitting while "theta" set to 0.5 to a (2nd order) Strang
    splitting.

    *Arguments*
      model (:py:class:`xalbrain.cardiacmodels.Model`)
        a Model object describing the simulation set-up
      parameters (:py:class:`dolfin.Parameters`, optional)
        a Parameters object controlling solver parameters

    *Example of usage*::

      from xalbrain import *

      # Describe the cardiac model
      mesh = UnitSquareMesh(100, 100)
      time = Constant(0.0)
      cell_model = FitzHughNagumoManual()
      stimulus = Expression("10*t*x[0]", t=time, degree=1)
      cm = Model(mesh, time, 1.0, 1.0, cell_model, stimulus)

      # Extract default solver parameters
      ps = SplittingSolver.default_parameters()

      # Examine the default parameters
      info(ps, True)

      # Update parameter dictionary
      ps["theta"] = 1.0 # Use first order splitting
      ps["CardiacODESolver"]["scheme"] = "GRL1" # Use Generalized Rush-Larsen scheme

      ps["pde_solver"] = "monodomain"                         # Use monodomain equations as the PDE model
      ps["MonodomainSolver"]["linear_solver_type"] = "direct" # Use direct linear solver of the PDEs
      ps["MonodomainSolver"]["theta"] = 1.0                   # Use backward Euler for temporal discretization for the PDEs

      solver = SplittingSolver(cm, parameters=ps)

      # Extract the solution fields and set the initial conditions
      (vs_, vs, vur) = solver.solution_fields()
      vs_.assign(cell_model.initial_conditions())

      # Solve
      dt = 0.1
      T = 1.0
      interval = (0.0, T)
      for (timestep, fields) in solver.solve(interval, dt):
          (vs_, vs, vur) = fields
          # Do something with the solutions


    *Assumptions*
      * The cardiac conductivities do not vary in time

    """

    def __init__(
        self,
        model: Model,
        ode_timestep: float = None,
        parameters: df.Parameters = None
    ) -> None:
        self._parameters = self.default_parameters()
        if parameters is not None:
            self._parameters.update(parameters)

        super().__init__(model, ode_timestep)

    @staticmethod
    def default_parameters() -> df.Parameters:
        """
        Initialize and return a set of default parameters for the splitting solver.

        *Returns*
          The set of default parameters (:py:class:`dolfin.Parameters`)

        *Example of usage*::

          info(SplittingSolver.default_parameters(), True)
        """
        parameters = df.Parameters("SplittingSolver")
        parameters.add("theta", 0.5, 0, 1)
        parameters.add("apply_stimulus_current_to_pde", False)
        # parameters.add("pde_solver", "bidomain", {"bidomain", "monodomain"})
        parameters.add("pde_solver", "bidomain")
        parameters.add(
            "ode_solver_choice",
            "CardiacODESolver"
        )


        # Add default parameters from ODE solver
        ode_solver_parameters = CardiacODESolver.default_parameters()
        ode_solver_parameters["scheme"] = "BDF1"
        parameters.add(ode_solver_parameters)

        # Add default parameters from ODE solver
        basic_ode_solver_parameters = BasicCardiacODESolver.default_parameters()
        parameters.add(basic_ode_solver_parameters)

        pde_solver_parameters = BidomainSolver.default_parameters()
        pde_solver_parameters["polynomial_degree"] = 1
        parameters.add(pde_solver_parameters)

        pde_solver_parameters = MonodomainSolver.default_parameters()
        pde_solver_parameters["polynomial_degree"] = 1
        parameters.add(pde_solver_parameters)
        return parameters

    def _create_ode_solver(self) -> tp.Union[BasicCardiacODESolver, CardiacODESolver]:
        """
        Helper function to initialize a suitable ODE solver from the cardiac model.
        """
        # Extract cardiac cell model from cardiac model
        cell_model = self._model.cell_models

        # Extract stimulus from the cardiac model(!)
        if self._parameters["apply_stimulus_current_to_pde"]:
            stimulus = None
        else:
            stimulus = self._model.stimulus()

        Solver = eval(self._parameters["ode_solver_choice"])
        parameters = self._parameters[Solver.__name__]

        solver = Solver(
            self._domain,
            self._time,
            cell_model,
            I_s=stimulus,
            parameters=parameters
        )
        return solver

    def _create_pde_solver(self) -> tp.Union[
            BidomainSolver,
            MonodomainSolver
    ]:
        """Helper function to initialize a suitable PDE solver from the cardiac model."""
        # Extract applied current from the cardiac model (stimulus invoked in the ODE step)
        applied_current = self._model.applied_current()
        ect_current = self._model.ect_current

        # Extract stimulus from the cardiac model
        if self._parameters["apply_stimulus_current_to_pde"]:
            stimulus = self._model.stimulus()
        else:
            stimulus = None

        # Extract conductivities from the cardiac model
        Mi, Me = self._model.conductivities()

        if self._parameters["pde_solver"] == "bidomain":
            PDESolver = BidomainSolver
            parameters = self._parameters["BidomainSolver"]
            parameters["theta"] = self._parameters["theta"]
            pde_args = (self._domain, self._time, Mi, Me)
            pde_kwargs = dict(
                I_s = stimulus,
                I_a = applied_current,
                ect_current = ect_current,
                v_ = self.vs[0],
                cell_domains = self._model.cell_domains,
                facet_domains = self._model.facet_domains,
                dirichlet_bc = self._model.dirichlet_bc_u,        # dirichlet_bc
                dirichlet_bc_v = self._model.dirichlet_bc_v,        # dirichlet_bc
                parameters = parameters
            )
        else:
            PDESolver = MonodomainSolver
            parameters = self._parameters["MonodomainSolver"]
            pde_args = (self._domain, self._time, Mi)
            pde_kwargs = dict(
                I_s = stimulus,
                v_ = self.vs[0],
                cell_domains = self._model.cell_domains,
                facet_domains = self._model.facet_domains,
                parameters = parameters
            )

        solver = PDESolver(*pde_args, **pde_kwargs)
        return solver


class MultiCellSplittingSolver(SplittingSolver):
    def __init__(
        self,
        model: Model,
        valid_cell_tags: tp.Sequence[int],
        parameter_map: "ODEMap",
        ode_timestep: float = None,
        parameters: df.Parameters = None
    ) -> None:
        self._parameters = self.default_parameters()
        if parameters is not None:
            self._parameters.update(parameters)

        self._cell_tags = valid_cell_tags       # cell tags in cell_function checked in ode solver
        self._parameter_map = parameter_map
        self._indicator_function = model.indicator_function
        super().__init__(model, ode_timestep)       # Must be called last

    @staticmethod
    def default_parameters() -> df.Parameters:
        """
        Initialize and return a set of default parameters for the splitting solver.

        *Returns*
          The set of default parameters (:py:class:`dolfin.Parameters`)

        *Example of usage*::

          info(SplittingSolver.default_parameters(), True)
        """
        parameters = df.Parameters("SplittingSolver")
        parameters.add("theta", 0.5, 0, 1)
        parameters.add("apply_stimulus_current_to_pde", False)
        # parameters.add("pde_solver", "bidomain", {"bidomain", "monodomain"})
        parameters.add("pde_solver", "bidomain")

        # Add default parameters from ODE solver
        multicell_ode_solver_parameters = MultiCellSolver.default_parameters()
        parameters.add(multicell_ode_solver_parameters)

        pde_solver_parameters = BidomainSolver.default_parameters()
        pde_solver_parameters["polynomial_degree"] = 1
        parameters.add(pde_solver_parameters)
        return parameters

    def _create_ode_solver(self) -> MultiCellSolver:
        """Helper function to initialize a suitable ODE solver from the cardiac model."""
        # Extract cardiac cell model from cardiac model
        assert self._cell_function is not None      # TODO: deprecate?
        assert self._indicator_function is not None
        cell_model = self._model.cell_models

        solver = MultiCellSolver(
            time=self._time,
            mesh=self._domain,
            cell_model=cell_model,
            cell_function=self._cell_function,
            valid_cell_tags=self._cell_tags,
            parameter_map=self._parameter_map,
            indicator_function=self._indicator_function,
            parameters=self._parameters["MultiCellSolver"],
        )
        return solver
