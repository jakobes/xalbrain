"""
This module contains splitting solvers for CardiacModel objects. In
particular, the two classes

  * SplittingSolver
  * BasicSplittingSolver

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

__all__ = ["SplittingSolver", "BasicSplittingSolver"]

from dolfin import *
from dolfin_adjoint import *
from beatadjoint import CardiacModel
from beatadjoint.cellsolver import BasicCardiacODESolver
from beatadjoint.bidomainsolver import BidomainSolver
from beatadjoint.utils import join, state_space

class BasicSplittingSolver:
    """

    A non-optimised solver for the bidomain equations based on the
    operator splitting scheme described in Sundnes et al 2006, p. 78
    ff.

    The solver computes as solutions:

      * "vs" (:py:class:`dolfin.Function`) representing the solution
        for the transmembrane potential and any additional state
        variables, and
      * "u" (:py:class:`dolfin.Function`) representing the solution
        for the extracellular potential.
      * Internally, the object "vur" representing the transmembrane
        potential in combination with the extracellular potential and
        an additional Lagrange multiplier is also used.

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

        # Create ODE solver and extract solution fields
        self.ode_solver = self._create_ode_solver()
        (self.vs_, self.vs) = self.ode_solver.solution_fields()
        self.VS = self.vs.function_space()

        # Create PDE solver and extract solution fields
        self.pde_solver = self._create_pde_solver()
        (self.v_, self.vur) = self.pde_solver.solution_fields()

        # FIXME
        self.V = self.VS.sub(0).collapse()
        R = FunctionSpace(self._domain, "R", 0)
        self.VUR = MixedFunctionSpace([self.V, self.V, R])

        # (Internal) solution fields
        self.u = Function(self.VUR.sub(1).collapse(), name="u")

        #self.vs_ = Function(self.VS, name="vs_")
        #self.vs = Function(self.VS, name="vs")

        #self.pde_solver = BidomainSolver()

    def _create_ode_solver(self):
        """Helper function to initialize a suitable ODE solver from
        the cardiac model."""

        # Extract cardiac cell model from cardiac model
        cell_model = self._model.cell_model

        # Extract stimulus from the cardiac model(!)
        stimulus = self._model.stimulus

        # Extract ode solver parameters
        params = self.parameters["BasicCardiacODESolver"]
        solver = BasicCardiacODESolver(self._domain, cell_model.num_states(),
                                       cell_model.F, cell_model.I,
                                       I_s = stimulus, params=params)
        return solver

    def _create_pde_solver(self):
        """Helper function to initialize a suitable PDE solver from
        the cardiac model."""

        # Extract applied current from the cardiac model (stimulus
        # invoked in the ODE step)
        applied_current = self._model.applied_current

        # Extract conductivities from the cardiac model
        (M_i, M_e) = self._model.conductivities()

        # Extract ode solver parameters
        params = self.parameters["BidomainSolver"]
        solver = BidomainSolver(self._domain, M_i, M_e,
                                I_s=None, I_a=applied_current, v_ = self.vs_[0],
                                params=params)
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

        params = Parameters("SplittingSolver")
        params.add("enable_adjoint", False)
        params.add("theta", 0.5)
        params.add("default_timestep", 1.0)

        params.add("potential_polynomial_degree", 1)
        params.add("num_threads", 0)
        params.add("use_avg_u_constraint", True)

        # Add default parameters from ODE solver, but update for V
        # space
        ode_solver_params = BasicCardiacODESolver.default_parameters()
        ode_solver_params["V_polynomial_degree"] = 1
        ode_solver_params["V_polynomial_family"] = "CG"
        params.add(ode_solver_params)

        pde_solver_params = BidomainSolver.default_parameters()
        pde_solver_params["polynomial_degree"] = 1
        params.add(pde_solver_params)

        # FIXME: Add default parameters from PDE solver
        pde_solver_params = LinearVariationalSolver.default_parameters()
        params.add(pde_solver_params)
        return params

    def solution_fields(self):
        """
        Return tuple of previous and current solution objects.

        Modifying these will modify the solution objects of the solver
        and thus provides a way for setting initial conditions for
        instance.

        *Returns*
          (previous vs, current vs, current u) (:py:class:`tuple` of :py:class:`dolfin.Function`)
        """
        return (self.vs_, self.vs, self.u)

    def solve(self, interval, dt):
        """
        Solve the problem given by the model on a given time interval
        (t0, t1) with a given timestep dt and return generator for a
        tuple of the time, the current vs solution and the current u
        solution.

        *Arguments*
          interval (:py:class:`tuple`)
            The time interval for the solve given by (t0, t1)
          dt (int)
            The timestep for the solve

        *Returns*
          (timestep, current vs, current u) via (:py:class:`genexpr`)

        *Example of usage*::

          # Create generator
          solutions = solver.solve((0.0, 1.0), 0.1)

          # Iterate over generator (computes solutions as you go)
          for (t, vs, u) in solutions:
            # do something with the solutions

        """
        # Initial set-up
        (T0, T) = interval
        t0 = T0
        t1 = T0 + dt
        annotate = self.parameters["enable_adjoint"]

        # Step through time steps until at end time.
        if annotate:
            adj_start_timestep(t0)
        while (t1 <= T + DOLFIN_EPS):

            info_blue("Solving on t = (%g, %g)" % (t0, t1))
            timestep = (t0, t1)
            (vs, u) = self.step(timestep)
            self.vs.assign(vs, annotate=annotate)
            self.u.assign(u, annotate=False) # Not part of solution algorithm

            # Yield current solutions
            yield (timestep, self.vs, self.u)

            # FIXME: Break here as usual.

            # Update previous and timetime
            finished = (t0 + dt) >= T
            self.vs_.assign(self.vs, annotate=annotate)
            if annotate:
                adj_inc_timestep(time=t1, finished=finished)
            t0 = t1
            t1 = t0 + dt

    def step(self, interval):
        """
        Solve the problem given by the model on a given time interval
        (t0, t1) with timestep given by the interval length and given
        initial conditions, return the current vs and the current u
        solutions.

        *Arguments*
          interval (:py:class:`tuple`)
            The time interval for the solve given by (t0, t1)

        *Returns*
          (current vs, current u) (:py:class:`tuple` of :py:class:`dolfin.Function`)
        """

        # Extract some parameters for readability
        theta = self.parameters["theta"]
        annotate = self.parameters["enable_adjoint"]

        # Extract time domain
        (t0, t1) = interval
        dt = (t1 - t0)
        t = t0 + theta*dt

        # Compute tentative membrane potential and state (vs_star)
        begin("Tentative ODE step")
        self.ode_solver._step((t0, t))
        end()

        # Compute tentative potentials vu = (v, u)
        begin("PDE step")
        self.vs_.assign(self.vs) # Update self.vs_ (pde_step) operates
                                 # on this one. Is this really a good
                                 # idea?
        vur = self.pde_step((t0, t1))#, self.vs)
        end()

        # Merge (inverse of split) v and s_star: (needed for adjointability)
        begin("Merging step")
        v = split(vur)[0]
        #(v, u, r) = split(vur)
        (v_star, s_star) = split(self.vs)
        v_s_star = join((v, s_star), self.VS, annotate=annotate,
                        solver_type="cg")
        end()

        # If first order splitting, we are done:
        if theta == 1.0:
            return (v_s_star, vur.split()[1])

        # Otherwise, we do another ode_step:
        begin("Corrective ODE step")
        self.vs_.assign(v_s_star)
        self.ode_solver._step((t, t1))
        end()

        return (self.vs, vur.split()[1])

    def pde_step(self, interval):
        """
        Solve the PDE step of the splitting scheme for the problem
        given by the model on a given time interval (t0, t1) with
        timestep given by the interval length and given initial
        conditions, return the current vur solution.

        *Arguments*
          interval (:py:class:`tuple`)
            The time interval for the solve given by (t0, t1)
          vs\_ (:py:class:`dolfin.Function`)
            Initial conditions for vs for this interval

        *Returns*
          current vur (:py:class:`dolfin.Function`)
        """

        # Hack, not sure if this is a good design (previous value for
        # s should not be required as data)
        (v_, s_) = split(self.vs_)

        # Extract interval and time-step
        (t0, t1) = interval
        k_n = Constant(t1 - t0)
        theta = self.parameters["theta"]
        annotate = self.parameters["enable_adjoint"]

        # Extract conductivities from model
        M_i, M_e = self._model.conductivities()

        # Define variational formulation
        (v, u, l) = TrialFunctions(self.VUR)
        (w, q, lamda) = TestFunctions(self.VUR)

        Dt_v = (v - v_)/k_n
        theta_parabolic = (theta*inner(M_i*grad(v), grad(w))*dx
                           + (1.0 - theta)*inner(M_i*grad(v_), grad(w))*dx
                           + inner(M_i*grad(u), grad(w))*dx)
        theta_elliptic = (theta*inner(M_i*grad(v), grad(q))*dx
                          + (1.0 - theta)*inner(M_i*grad(v_), grad(q))*dx
                          + inner((M_i + M_e)*grad(u), grad(q))*dx)
        G = (Dt_v*w*dx + theta_parabolic + theta_elliptic + (lamda*u + l*q)*dx)

        # Add applied current as source in elliptic equation if
        # applicable
        if self._model.applied_current:
            t = t0 + theta*(t1 - t0)
            self._model.applied_current.t = t
            G -= self._model.applied_current*q*dx

        # Define variational problem
        a, L = system(G)
        vur = Function(self.VUR, name="pde_vur")
        pde = LinearVariationalProblem(a, L, vur)

        # Set-up solver
        solver = LinearVariationalSolver(pde)
        solver_params = self.parameters["linear_variational_solver"]
        solver.parameters.update(solver_params)

        # Solve system
        solver.solve(annotate=annotate)

        return vur

class SplittingSolver(BasicSplittingSolver):
    """

    An optimised solver for the bidomain equations based on the
    operator splitting scheme described in Sundnes et al 2006, p. 78
    ff.

    The solver computes as solutions:

      * "vs" (:py:class:`dolfin.Function`) representing the solution
        for the transmembrane potential and any additional state
        variables, and
      * "u" (:py:class:`dolfin.Function`) representing the solution
        for the extracellular potential.

    The algorithm can be controlled by a number of parameters. In
    particular, the splitting algorithm can be controlled by the
    parameter "theta": "theta" set to 1.0 corresponds to a (1st order)
    Godunov splitting while "theta" set to 0.5 to a (2nd order) Strang
    splitting.

    *Arguments*
      model (:py:class:`beatadjoint.cardiacmodels.CardiacModel`)
        a CardiacModel object describing the simulation set-up
      parameters (:py:class:`dolfin.Parameters`)
        a Parameters object controlling solver parameters

    *Assumptions*
      * The cardiac conductivities do not vary in time


    """

    def __init__(self, model, parameters=None):
        BasicSplittingSolver.__init__(self, model, parameters)

        # Define forms for pde_step
        self._k_n = Constant(self.parameters["default_timestep"])
        (self._a, self._L) = self.pde_variational_problem(self._k_n, self.vs_)

        # Pre-assemble left-hand side (will be updated if time-step
        # changes)
        self._A = assemble(self._a, annotate=True)

        # Tune solver types
        solver_parameters = self.parameters["linear_variational_solver"]
        solver_type = solver_parameters["linear_solver"]
        if solver_type == "direct":
            self._linear_solver = LUSolver(self._A)
            self._linear_solver.parameters.update(solver_parameters["lu_solver"])
        elif solver_type == "iterative":
            self._linear_solver = KrylovSolver("gmres", "amg")
            self._linear_solver.parameters.update(solver_parameters["krylov_solver"])
            self._linear_solver.set_operator(self._A)
        else:
            error("Unknown linear_pde_solver specified: %s" % solver_type)

    @staticmethod
    def default_parameters():
        """Initialize and return a set of default parameters for the
        splitting solver

        *Returns*
          A set of parameters (:py:class:`dolfin.Parameters`)

        To inspect all the default parameters, do::

          info(SplittingSolver.default_parameters(), True)
        """

        parameters = BasicSplittingSolver.default_parameters()

        # Customize linear solver parameters
        ps = parameters["linear_variational_solver"]
        ps["linear_solver"] = "iterative"
        ps["krylov_solver"]["preconditioner"]["same_nonzero_pattern"] = True
        ps["lu_solver"]["same_nonzero_pattern"] = True

        return parameters

    def linear_solver(self):
        """Return the linear solver (re-)used for the PDE step of the
        splitting algorithm.

        *Returns*
          linear solver (:py:class:`dolfin.LinearSolver`)
        """
        return self._linear_solver

    def pde_variational_problem(self, k_n, vs_):
        """Create and return the variational problem for the PDE step
        of the splitting algorithm.

        *Arguments*
          k_n (:py:class:`ufl.Expr` or float)
            The time step
          vs\_ (:py:class:`dolfin.Function`)
            Solution for vs at "previous time"

        *Returns*
          (lhs, rhs) (:py:class:`tuple` of :py:class:`ufl.Form`)
        """

        # Extract conductivities from model
        M_i, M_e = self._model.conductivities()

        # Extract theta parameter
        theta = self.parameters["theta"]

        # Define variational formulation
        use_R = self.parameters["use_avg_u_constraint"]
        if not use_R:
            self.VUR = MixedFunctionSpace([self.V, self.V])
            (v, u) = TrialFunctions(self.VUR)
            (w, q) = TestFunctions(self.VUR)
        else:
            (v, u, l) = TrialFunctions(self.VUR)
            (w, q, lamda) = TestFunctions(self.VUR)

        # Set-up variational problem
        (v_, s_) = split(vs_)
        Dt_v = (v - v_)

        theta_parabolic = (theta*inner(M_i*grad(v), grad(w))*dx
                           + (1.0 - theta)*inner(M_i*grad(v_), grad(w))*dx
                           + inner(M_i*grad(u), grad(w))*dx)
        theta_elliptic = (theta*inner(M_i*grad(v), grad(q))*dx
                          + (1.0 - theta)*inner(M_i*grad(v_), grad(q))*dx
                          + inner((M_i + M_e)*grad(u), grad(q))*dx)

        G = (Dt_v*w*dx + k_n*theta_parabolic + theta_elliptic)

        if use_R:
            G += (lamda*u + l*q)*dx

        # Add applied current if specified
        if self._model.applied_current:
            G -= k_n*self._model.applied_current*w*dx

        (a, L) = system(G)
        return (a, L)

    def pde_step(self, interval):
        """
        Solve the PDE step of the splitting scheme for the problem
        given by the model on a given time interval (t0, t1) with
        timestep given by the interval length and given initial
        conditions, return the current vur solution.

        *Arguments*
          interval (:py:class:`tuple`)
            The time interval for the solve given by (t0, t1)
          vs\_ (:py:class:`dolfin.Function`)
            Initial conditions for vs for this interval

        *Returns*
          current vur (:py:class:`dolfin.Function`)

        """

        # Extract interval and time-step
        (t0, t1) = interval
        dt = (t1 - t0)
        theta = self.parameters["theta"]
        t = t0 + theta*dt
        annotate = self.parameters["enable_adjoint"]

        # Update previous solution
        #self.vs_.assign(vs_, annotate=annotate)

        # Assemble left-hand-side: only re-assemble if necessary, and
        # reuse all solver data possible
        solver = self.linear_solver()
        tolerance = 1.e-12
        if abs(dt - float(self._k_n)) < tolerance:
            A = self._A
            if isinstance(solver, LUSolver):
                info("Reusing LU factorization")
                solver.parameters["reuse_factorization"] = True
            elif isinstance(solver, KrylovSolver):
                info("Reusing KrylovSolver preconditioner")
                solver.parameters["preconditioner"]["reuse"] = True
            else:
                pass
        else:
            self._k_n.assign(Constant(dt))#, annotate=annotate) # FIXME
            A = assemble(self._a, annotate=True)
            self._A = A
            solver.set_operator(self._A)
            if isinstance(solver, LUSolver):
                solver.parameters["reuse_factorization"] = False
            elif isinstance(solver, KrylovSolver):
                solver.parameters["preconditioner"]["reuse"] = False
            else:
                pass

        # Assemble right-hand-side
        if self._model.applied_current:
            self._model.applied_current.t = t
        b = assemble(self._L, annotate=True)

        # Solve system
        vur = Function(self.VUR, name="pde_vur")
        solver.solve(vur.vector(), b, annotate=annotate)

        # Rescale u if KrylovSolver is used...
        if (isinstance(solver, KrylovSolver) and annotate==False):
            info_blue("Normalizing u")
            avg_u = assemble(split(vur)[1]*dx)
            bar = project(Constant((0.0, avg_u, 0.0)), self.VUR)
            vur.vector().axpy(-1.0, bar.vector())

        return vur

