"""This module contains splitting solvers for CardiacModel objects. 
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
:py:class:`~xalbrain.cardiacmodels.CardiacModel` providing the
required input specification of the problem. In particular, the
applied current :math:`I_a` is extracted from the
:py:attr:`~xalbrain.cardiacmodels.CardiacModel.applied_current`
attribute, while the stimulus :math:`I_s` is extracted from the
:py:attr:`~xalbrain.cardiacmodels.CardiacModel.stimulus` attribute.

It should be possible to use the solvers interchangably. However, note
that the BasicSplittingSolver is not optimised and should be used for
testing or debugging purposes primarily.
"""

# Copyright (C) 2012-2013 Marie E. Rognes (meg@simula.no)
# Use and modify at will
# Last changed: 2013-04-15

__all__ = ["SplittingSolver", "BasicSplittingSolver",]

import dolfin as df
import numpy as np

from xalbrain import CardiacModel

from xalbrain.cellsolver import (
    BasicCardiacODESolver,
    CardiacODESolver,
)

from xalbrain.bidomainsolver import (
    BasicBidomainSolver,
    BidomainSolver,
)

from xalbrain.monodomainsolver import (
    BasicMonodomainSolver,
    MonodomainSolver,
)

from xalbrain.utils import (
    state_space,
    TimeStepper,
)

from typing import (
    Any,
    Tuple,
    Generator,
    Union,
)


class BasicSplittingSolver:
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
      model (:py:class:`xalbrain.cardiacmodels.CardiacModel`)
        a CardiacModel object describing the simulation set-up
      params (:py:class:`dolfin.Parameters`, optional)
        a Parameters object controlling solver parameters

    *Assumptions*
      * The cardiac conductivities do not vary in time
    """

    def __init__(
            self,
            model: CardiacModel,
            ode_timestep: float = None,
            params: df.Parameters = None
    ) -> None:
        """Create solver from given Cardiac Model and (optional) parameters."""
        assert isinstance(model, CardiacModel), "Expecting CardiacModel as first argument"

        self._ode_timestep = ode_timestep

        # Set model and parameters
        self._model = model
        self._parameters = self.default_parameters()
        if params is not None:
            self._parameters.update(params)

        # Extract solution domain
        self._domain = self._model.mesh
        self._time = self._model.time

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

    def _create_ode_solver(self):
        """Helper function to initialize a suitable ODE solver from the cardiac model."""
        cell_model = self._model.cell_models

        # Extract stimulus from the cardiac model(!)
        if self._parameters["apply_stimulus_current_to_pde"]:
            stimulus = self._model.stimulus()
        else:
            stimulus = None

        params = self._parameters["BasicCardiacODESolver"]
        solver = BasicCardiacODESolver(
            self._domain,
            self._time,
            cell_model,
            I_s=stimulus,
            params=params
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
            params = self._parameters["BasicBidomainSolver"]
            params["theta"] = self._parameters["theta"]
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
                params = params
            )
        else:
            PDESolver = BasicMonodomainSolver
            params = self._parameters["BasicMonodomainSolver"]
            pde_args = (self._domain, self._time, Mi)
            pde_kwargs = dict(
                I_s = stimulus,
                v_ = self.vs[0],
                params = params,
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
        params = df.Parameters("BasicSplittingSolver")
        params.add("enable_adjoint", False)
        params.add("theta", 0.5, 0., 1.)
        params.add("apply_stimulus_current_to_pde", False)
        params.add("pde_solver", "bidomain")

        # Add default parameters from ODE solver, but update for V space
        ode_solver_params = BasicCardiacODESolver.default_parameters()
        ode_solver_params["V_polynomial_degree"] = 1
        ode_solver_params["V_polynomial_family"] = "CG"
        params.add(ode_solver_params)

        pde_solver_params = BasicBidomainSolver.default_parameters()
        pde_solver_params["polynomial_degree"] = 1
        params.add(pde_solver_params)

        pde_solver_params = BasicMonodomainSolver.default_parameters()
        pde_solver_params["polynomial_degree"] = 1
        params.add(pde_solver_params)
        return params

    def solution_fields(self) -> Tuple[df.Function, df.Function, df.Function]:
        """
        Return tuple of previous and current solution objects.

        Modifying these will modify the solution objects of the solver
        and thus provides a way for setting initial conditions for
        instance.

        *Returns*
          (previous vs, current vs, current vur) (:py:class:`tuple` of :py:class:`dolfin.Function`)
        """
        return self.vs_, self.vs, self.vur

    def solve(self, interval, dt) -> Generator[Tuple[Tuple[float, float], df.Function], None, None]:
        """
        Solve the problem given by the model on a time interval with a given time step.
        Return a generator for a tuple of the time step and the solution fields.

        *Arguments*
          interval (:py:class:`tuple`)
            The time interval for the solve given by (t0, t1)
          dt (int, list of tuples of floats)
            The timestep for the solve. A list of tuples of floats can
            also be passed. Each tuple should contain two floats where the
            first includes the start time and the second the dt.

        *Returns*
          (timestep, solution_fields) via (:py:class:`genexpr`)

        *Example of usage*::

          # Create generator
          dts = [(0., 0.1), (1.0, 0.05), (2.0, 0.1)]
          solutions = solver.solve((0.0, 1.0), dts)

          # Iterate over generator (computes solutions as you go)
          for ((t0, t1), (vs_, vs, vur)) in solutions:
            # do something with the solutions

        """
        # Create timestepper
        time_stepper = TimeStepper(interval, dt)

        for t0, t1 in time_stepper:
            self.step((t0, t1))

            # Yield solutions
            yield (t0, t1), self.solution_fields()

            # Update previous solution
            self.vs_.assign(self.vs)

    def step(self, interval: Tuple[float, float]) -> None:
        """
        Solve the pde for one time step.

        *Arguments*
          interval (:py:class:`tuple`)
            The time interval for the solve given by (t0, t1)

        *Invariants*
          Given self._vs in a correct state at t0, provide v and s (in
          self.vs) and u (in self.vur) in a correct state at t1. (Note
          that self.vur[0] == self.vs[0] only if theta = 1.0.)
        """
        theta = self._parameters["theta"]

        # Extract time domain
        t0, t1 = interval
        dt = t1 - t0
        t = t0 + theta*dt

        # Compute tentative membrane potential and state (vs_star)
        # df.begin(df.PROGRESS, "Tentative ODE step")
        # Assumes that its vs_ is in the correct state, gives its vs
        # in the current state
        # self.ode_solver.step((t0, t))
        if self._ode_timestep is None:
            self.ode_solver.step((t0, t))
        else:
            # Take multiple ODE steps for each pde step
            for _ in self.ode_solver.solve((t0, t), self._ode_timestep):
                pass

        self.vs_.assign(self.vs)

        # Compute tentative potentials vu = (v, u)
        # Assumes that its vs_ is in the correct state, gives vur in
        # the current state
        self.pde_solver.step((t0, t1))

        # If first order splitting, we need to ensure that self.vs is
        # up to date, but otherwise we are done.
        if theta == 1.0:
            # Assumes that the v part of its vur and the s part of its
            # vs are in the correct state, provides input argument(in
            # this case self.vs) in its correct state
            self.merge(self.vs)
            return

        # Otherwise, we do another ode_step:

        # Assumes that the v part of its vur and the s part of its vs
        # are in the correct state, provides input argument (in this
        # case self.vs_) in its correct state
        self.merge(self.vs_)    # self.vs_.sub(0) <- self.vur.sub(0)
        # Assumes that its vs_ is in the correct state, provides vs in the correct state

        # self.ode_solver.step((t0, t))
        if self._ode_timestep is None:
            self.ode_solver.step((t0, t))
        else:
            # Take multiple ODE steps for each pde step
            for _ in self.ode_solver.solve((t0, t), self._ode_timestep):
                pass

    def merge(self, solution: df.Function) -> None:
        """
        Combine solutions from the PDE and the ODE to form a single mixed function.

        *Arguments*
          solution (:py:class:`dolfin.Function`)
            Function holding the combined result
        """
        timer = df.Timer("Merge step")
        if self._parameters["pde_solver"] == "bidomain":
            v = self.vur.sub(0)
        else:
            v = self.vur
        self.merger.assign(solution.sub(0), v)
        timer.stop()

    @property
    def model(self) -> CardiacModel:
        """Return the brain."""
        return self._model

    @property
    def parameters(self) -> df.Parameters:
        """Return the parameters."""
        return self._parameters


class SplittingSolver(BasicSplittingSolver):
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
      model (:py:class:`xalbrain.cardiacmodels.CardiacModel`)
        a CardiacModel object describing the simulation set-up
      params (:py:class:`dolfin.Parameters`, optional)
        a Parameters object controlling solver parameters

    *Example of usage*::

      from xalbrain import *

      # Describe the cardiac model
      mesh = UnitSquareMesh(100, 100)
      time = Constant(0.0)
      cell_model = FitzHughNagumoManual()
      stimulus = Expression("10*t*x[0]", t=time, degree=1)
      cm = CardiacModel(mesh, time, 1.0, 1.0, cell_model, stimulus)

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

      solver = SplittingSolver(cm, params=ps)

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
            model: CardiacModel,
            ode_timestep: float = None,
            params: df.parameters = None
    ) -> None:
        super().__init__(model, ode_timestep, params)

    @staticmethod
    def default_parameters() -> df.Parameters:
        """
        Initialize and return a set of default parameters for the splitting solver.

        *Returns*
          The set of default parameters (:py:class:`dolfin.Parameters`)

        *Example of usage*::

          info(SplittingSolver.default_parameters(), True)
        """
        params = df.Parameters("SplittingSolver")
        params.add("enable_adjoint", False)
        params.add("theta", 0.5, 0, 1)
        params.add("apply_stimulus_current_to_pde", False)
        # params.add("pde_solver", "bidomain", {"bidomain", "monodomain"})
        params.add("pde_solver", "bidomain")
        params.add(
            "ode_solver_choice",
            "CardiacODESolver"
        )

        # Add default parameters from ODE solver
        ode_solver_params = CardiacODESolver.default_parameters()
        ode_solver_params["scheme"] = "BDF1"
        params.add(ode_solver_params)

        # Add default parameters from ODE solver
        basic_ode_solver_params = BasicCardiacODESolver.default_parameters()
        basic_ode_solver_params["V_polynomial_degree"] = 1
        basic_ode_solver_params["V_polynomial_family"] = "CG"
        params.add(basic_ode_solver_params)

        pde_solver_params = BidomainSolver.default_parameters()
        pde_solver_params["polynomial_degree"] = 1
        params.add(pde_solver_params)

        pde_solver_params = MonodomainSolver.default_parameters()
        pde_solver_params["polynomial_degree"] = 1
        params.add(pde_solver_params)
        return params

    def _create_ode_solver(self) -> Union[BasicCardiacODESolver, CardiacODESolver]:
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
        params = self._parameters[Solver.__name__]

        solver = Solver(
            self._domain,
            self._time,
            cell_model,
            I_s=stimulus,
            params=params
        )
        return solver

    def _create_pde_solver(self) -> Union[
            BasicBidomainSolver,
            BidomainSolver,
            BasicMonodomainSolver,
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
            params = self._parameters["BidomainSolver"]
            params["theta"] = self._parameters["theta"]
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
                params = params
            )
        else:
            PDESolver = MonodomainSolver
            params = self._parameters["MonodomainSolver"]
            pde_args = (self._domain, self._time, Mi)
            pde_kwargs = dict(
                I_s = stimulus,
                v_ = self.vs[0],
                cell_domains = self._model.cell_domains,
                facet_domains = self._model.facet_domains,
                params = params
            )

        solver = PDESolver(*pde_args, **pde_kwargs)
        return solver
