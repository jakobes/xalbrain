"""
This module contains splitting solvers for CardiacModel objects. In
particular, the classes

  * BasicSplittingSolver
  * SplittingSolver

These solvers solve the bidomain equations on the form: find the
transmembrane potential :math:`v = v(x, t)`, the extracellular
potential :math:`u = u(x, t)`, and any additional state variables
:math:`s = s(x, t)` such that

.. math::

   v_t - \mathrm{div} ( G_i v + G_i u) = - I_{ion}(v, s) + I_s

   \mathrm{div} (G_i v + (G_i + G_e) u) = I_a

   s_t = F(v, s)

where the subscript :math:`t` denotes the time derivative;
:math:`G_x` denotes a weighted gradient: :math:`G_x = M_x
\mathrm{grad}(v)` for :math:`x \in \{i, e\}`, where :math:`M_i` and
:math:`M_e` are cardiac conductivity tensors; :math:`I_s` and
:math:`I_a` are prescribed input; :math:`I_{ion}` and :math:`F` are
given functions specified by a cell model (or alternatively defining
the cell model). In addition, initial conditions are given for
:math:`v` and :math:`s`:

.. math::

   v(x, 0) = v_0

   s(x, 0) = s_0

Finally, boundary conditions must be prescribed. These solvers assume
pure Neumann boundary conditions for :math:`v` and :math:`u` and
enforce the additional average value zero constraint for u.

The solvers take as input a
:py:class:`~beatadjoint.cardiacmodels.CardiacModel` providing the
required input specification of the problem. In particular, the
applied current :math:`I_a` is extracted from the
:py:attr:`~beatadjoint.cardiacmodels.CardiacModel.applied_current`
attribute, while the stimulus :math:`I_s` is extracted from the
:py:attr:`~beatadjoint.cardiacmodels.CardiacModel.stimulus` attribute.

It should be possible to use the solvers interchangably. However, note
that the BasicSplittingSolver is not optimised and should be used for
testing or debugging purposes primarily.
"""

# Copyright (C) 2012-2013 Marie E. Rognes (meg@simula.no)
# Use and modify at will
# Last changed: 2013-04-15

__all__ = ["BasicSplittingSolver", "SplittingSolver"]

from dolfin import *
import dolfin
from dolfin_adjoint import *
from beatadjoint import CardiacModel
from beatadjoint.cellsolver import BasicCardiacODESolver, CardiacODESolver
from beatadjoint.bidomainsolver import BasicBidomainSolver, BidomainSolver
from beatadjoint.monodomainsolver import BasicMonodomainSolver, MonodomainSolver
from beatadjoint.utils import state_space, TimeStepper, Projecter

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
    :py:class:`beatadjoint.splittingsolver.SplittingSolver`.

    *Arguments*
      model (:py:class:`beatadjoint.cardiacmodels.CardiacModel`)
        a CardiacModel object describing the simulation set-up
      parameters (:py:class:`dolfin.Parameters`, optional)
        a Parameters object controlling solver parameters

    *Assumptions*
      * The cardiac conductivities do not vary in time

    """
    def __init__(self, model, params=None):
        "Create solver from given Cardiac Model and (optional) parameters."

        assert isinstance(model, CardiacModel), \
            "Expecting CardiacModel as first argument"

        # Set model and parameters
        self._model = model
        self.parameters = self.default_parameters()
        if params is not None:
            self.parameters.update(params)
        
        # Extract solution domain
        self._domain = self._model.domain
        self._time = self._model.time

        # Create ODE solver and extract solution fields
        self.ode_solver = self._create_ode_solver()
        (self.vs_, self.vs) = self.ode_solver.solution_fields()
        self.VS = self.vs.function_space()

        # Create PDE solver and extract solution fields
        self.pde_solver = self._create_pde_solver()
        (self.v_, self.vur) = self.pde_solver.solution_fields()

        # If not enable adjoint
        if not self.parameters.enable_adjoint:
            parameters.adjoint.record_all = False
            parameters.adjoint.stop_annotating = False

    def _create_ode_solver(self):
        """Helper function to initialize a suitable ODE solver from
        the cardiac model."""

        # Extract cardiac cell model from cardiac model
        cell_model = self._model.cell_model

        # Extract stimulus from the cardiac model(!)
        if self.parameters.apply_stimulus_current_to_pde:
            stimulus = None
        else:
            stimulus = self._model.stimulus

        # Extract ode solver parameters
        params = self.parameters["BasicCardiacODESolver"]
        solver = BasicCardiacODESolver(self._domain, self._time,
                                       cell_model.num_states(),
                                       cell_model.F, cell_model.I,
                                       I_s=stimulus, params=params)
        return solver

    def _create_pde_solver(self):
        """Helper function to initialize a suitable PDE solver from
        the cardiac model."""

        # Extract applied current from the cardiac model (stimulus
        # invoked in the ODE step)
        applied_current = self._model.applied_current

        # Extract stimulus from the cardiac model(!)
        if self.parameters.apply_stimulus_current_to_pde:
            stimulus = self._model.stimulus
        else:
            stimulus = None

        # Extract conductivities from the cardiac model
        (M_i, M_e) = self._model.conductivities()

        if self.parameters["pde_solver"] == "bidomain":
            PDESolver = BasicBidomainSolver
            params = self.parameters["BasicBidomainSolver"]
            args = (self._domain, self._time, M_i, M_e)
            kwargs = dict(I_s=stimulus, I_a=applied_current,
                          v_=self.vs[0], params=params)
        else:
            PDESolver = BasicMonodomainSolver
            params = self.parameters["BasicMonodomainSolver"]
            args = (self._domain, self._time, M_i,)
            kwargs = dict(I_s=stimulus, v_=self.vs[0], params=params)
        
        # Propagate enable_adjoint to Bidomain solver
        if params.has_key("enable_adjoint"):
            params["enable_adjoint"] = self.parameters["enable_adjoint"]

        solver = PDESolver(*args, **kwargs)

        return solver

    @staticmethod
    def default_parameters():
        """Initialize and return a set of default parameters for the
        splitting solver

        *Returns*
          A set of parameters (:py:class:`dolfin.Parameters`)

        To inspect all the default parameters, do::

          info(BasicSplittingSolver.default_parameters(), True)
        """

        params = Parameters("BasicSplittingSolver")
        params.add("enable_adjoint", True)
        params.add("theta", 0.5)
        params.add("apply_stimulus_current_to_pde", False)
        params.add("pde_solver", "bidomain", ["bidomain", "monodomain"])

        # Add default parameters from ODE solver, but update for V
        # space
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

    def solution_fields(self):
        """
        Return tuple of previous and current solution objects.

        Modifying these will modify the solution objects of the solver
        and thus provides a way for setting initial conditions for
        instance.

        *Returns*
          (previous vs, current vs, current vur) (:py:class:`tuple` of :py:class:`dolfin.Function`)
        """
        return (self.vs_, self.vs, self.vur)

    def solve(self, interval, dt):
        """
        Solve the problem given by the model on a given time interval
        (t0, t1) with a given timestep dt and return generator for a
        tuple of the time step and the solution fields.

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
        time_stepper = TimeStepper(interval, dt, \
                                   annotate=self.parameters["enable_adjoint"])

        for t0, t1 in time_stepper:

            info_blue("Solving on t = (%g, %g)" % (t0, t1))
            self.step((t0, t1))

            # Yield solutions
            yield (t0, t1), self.solution_fields()

            # FIXME: This eventually breaks in parallel!?
            self.vs_.assign(self.vs)

    def step(self, interval):
        """
        Solve the problem given by the model on a given time interval
        (t0, t1) with timestep given by the interval length.

        *Arguments*
          interval (:py:class:`tuple`)
            The time interval for the solve given by (t0, t1)

        *Invariants*
          Given self._vs in a correct state at t0, provide v and s (in
          self.vs) and u (in self.vur) in a correct state at t1. (Note
          that self.vur[0] == self.vs[0] only if theta = 1.0.)
        """

        # Extract some parameters for readability
        theta = self.parameters["theta"]

        # Extract time domain
        (t0, t1) = interval
        dt = (t1 - t0)
        t = t0 + theta*dt

        # Compute tentative membrane potential and state (vs_star)
        begin("Tentative ODE step")
        # Assumes that its vs_ is in the correct state, gives its vs
        # in the current state
        self.ode_solver.step((t0, t))
        end()

        # Compute tentative potentials vu = (v, u)
        begin("PDE step")
        # Assumes that its vs_ is in the correct state, gives vur in
        # the current state
        self.pde_solver.step((t0, t1))
        end()

        # If first order splitting, we need to ensure that self.vs is
        # up to date, but otherwise we are done.
        if theta == 1.0:
            # Assumes that the v part of its vur and the s part of its
            # vs are in the correct state, provides input argument(in
            # this case self.vs) in its correct state
            self.merge(self.vs)
            return

        # Otherwise, we do another ode_step:
        begin("Corrective ODE step")

        # Assumes that the v part of its vur and the s part of its vs
        # are in the correct state, provides input argument (in this
        # case self.vs_) in its correct state
        self.merge(self.vs_)

        # Assumes that its vs_ is in the correct state, provides vs in
        # the correct state
        self.ode_solver.step((t, t1))
        end()

    def merge(self, solution):
        """
        Combine solutions from the PDE solve and the ODE solve to form
        a single mixed function.

        *Arguments*
          solution (:py:class:`dolfin.Function`)
            Function holding the combined result
        """
        begin("Merging")

        if self.parameters["pde_solver"] == "bidomain":
            v = split(self.vur)[0]
        else:
            v = self.vur
        
        s = split(self.vs)[1]

        if len(s.shape())==1:
            proj = as_vector([v]+[s[i] for i in range(s.shape()[0])])
        else:
            proj = as_vector((v, s))

        VS = self.vs.function_space()
        p = TrialFunction(VS)
        q = TestFunction(VS)
        a = inner(p, q)*dx()
        L = inner(proj, q)*dx() # FIXME: Shape mismatch?
        solve(a == L, solution, solver_parameters={"linear_solver": "cg"})
        end()

class SplittingSolver(BasicSplittingSolver):

    def __init__(self, model, params=None):
        BasicSplittingSolver.__init__(self, model, params)

        # Set-up projection solver (for optimised merging) of fields
        self.vs_projecter = Projecter(self.VS,
                                      params=self.parameters["Projecter"])

    @staticmethod
    def default_parameters():
        """Initialize and return a set of default parameters for the
        splitting solver

        *Returns*
          A set of parameters (:py:class:`dolfin.Parameters`)

        To inspect all the default parameters, do::

          info(SplittingSolver.default_parameters(), True)
        """

        params = Parameters("SplittingSolver")
        params.add("enable_adjoint", True)
        params.add("theta", 0.5)
        params.add("apply_stimulus_current_to_pde", False)
        params.add("pde_solver", "bidomain", ["bidomain", "monodomain"])

        # Add default parameters from ODE solver
        ode_solver_params = CardiacODESolver.default_parameters()
        ode_solver_params["scheme"] = "CN2"
        params.add(ode_solver_params)

        pde_solver_params = BidomainSolver.default_parameters()
        pde_solver_params["polynomial_degree"] = 1
        params.add(pde_solver_params)

        pde_solver_params = MonodomainSolver.default_parameters()
        pde_solver_params["polynomial_degree"] = 1
        params.add(pde_solver_params)

        projecter_params = Projecter.default_parameters()
        params.add(projecter_params)

        return params

    def _create_ode_solver(self):
        """Helper function to initialize a suitable ODE solver from
        the cardiac model."""

        # Extract cardiac cell model from cardiac model
        cell_model = self._model.cell_model

        # Extract stimulus from the cardiac model(!)
        if self.parameters.apply_stimulus_current_to_pde:
            stimulus = None
        else:
            stimulus = self._model.stimulus

        # Extract ode solver parameters
        params = self.parameters["CardiacODESolver"]
        solver = CardiacODESolver(self._domain, self._time,
                                  cell_model.num_states(),
                                  cell_model.F, cell_model.I,
                                  I_s=stimulus, params=params)
        return solver

    def _create_pde_solver(self):
        """Helper function to initialize a suitable PDE solver from
        the cardiac model."""

        # Extract applied current from the cardiac model (stimulus
        # invoked in the ODE step)
        applied_current = self._model.applied_current

        # Extract stimulus from the cardiac model(!)
        if self.parameters.apply_stimulus_current_to_pde:
            stimulus = self._model.stimulus
        else:
            stimulus = None

        # Extract conductivities from the cardiac model
        (M_i, M_e) = self._model.conductivities()

        if self.parameters["pde_solver"] == "bidomain":
            PDESolver = BidomainSolver
            params = self.parameters["BidomainSolver"]
            args = (self._domain, self._time, M_i, M_e)
            kwargs = dict(I_s=stimulus, I_a=applied_current,
                          v_=self.vs[0], params=params)
        else:
            PDESolver = MonodomainSolver
            params = self.parameters["MonodomainSolver"]
            args = (self._domain, self._time, M_i,)
            kwargs = dict(I_s=stimulus, v_=self.vs[0], params=params)
        
        # Propagate enable_adjoint to Bidomain solver
        if params.has_key("enable_adjoint"):
            params["enable_adjoint"] = self.parameters["enable_adjoint"]

        solver = PDESolver(*args, **kwargs)

        return solver

    def merge(self, solution):
        """
        Combine solutions from the PDE solve and the ODE solve to form
        a single mixed function.

        *Arguments*
          solution (:py:class:`dolfin.Function`)
            Function holding the combined result
        """
        # Disabled for now (does not pass replay)

        begin("Merging using custom projecter")
        if self.parameters["pde_solver"] == "bidomain":
            v = split(self.vur)[0]
        else:
            v = self.vur
        
        s = split(self.vs)[1]
        # FIXME: We should not need to do a projection. A sub function assign would
        # FIXME: be sufficient.
        if len(s.shape())==1:
            self.vs_projecter(as_vector([v]+[s[i] for i in range(s.shape()[0])]), solution)
        else:
            self.vs_projecter(as_vector((v, s)), solution)

        end()
