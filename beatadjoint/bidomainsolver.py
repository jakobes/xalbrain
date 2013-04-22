"""
This solver solves the (pure) bidomain equations on the form: find the
transmembrane potential :math:`v = v(x, t)` and the extracellular
potential :math:`u = u(x, t)` such that

.. math::

   v_t - \mathrm{div} ( G_i v + G_i u) = I_s

   \mathrm{div} (G_i v + (G_i + G_e) u) = I_a

where the subscript :math:`t` denotes the time derivative; :math:`G_x`
denotes a weighted gradient: :math:`G_x = M_x \mathrm{grad}(v)` for
:math:`x \in \{i, e\}`, where :math:`M_i` and :math:`M_e` are the
intracellular and extracellular cardiac conductivity tensors,
respectively; :math:`I_s` and :math:`I_a` are prescribed input. In
addition, initial conditions are given for :math:`v`:

.. math::

   v(x, 0) = v_0

Finally, boundary conditions must be prescribed. For now, this solver
assumes pure homogeneous Neumann boundary conditions for :math:`v` and
:math:`u` and enforces the additional average value zero constraint
for u.

"""

# Copyright (C) 2013 Marie E. Rognes (meg@simula.no)
# Use and modify at will
# Last changed: 2013-04-18

__all__ = ["BasicBidomainSolver", "BidomainSolver"]

from dolfin import *
from dolfin_adjoint import *
from beatadjoint.utils import end_of_time

class BasicBidomainSolver(object):
    """This solver is based on a theta-scheme discretization in time
    and CG_1 x CG_1 (x R) elements in space.

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

      M_i (:py:class:`ufl.Expr`)
        The intracellular conductivity tensor (as an UFL expression)

      M_e (:py:class:`ufl.Expr`)
        The extracellular conductivity tensor (as an UFL expression)

      I_s (:py:class:`dolfin.Expression`, optional)
        A (typically time-dependent) external stimulus

      I_a (:py:class:`dolfin.Expression`, optional)
        A (typically time-dependent) external applied current

      v\_ (:py:class:`ufl.Expr`, optional)
        Initial condition for v. A new :py:class:`dolfin.Function`
        will be created if none is given.

      params (:py:class:`dolfin.Parameters`, optional)
        Solver parameters

      """
    def __init__(self, domain, M_i, M_e, I_s=None, I_a=None, v_=None,
                 params=None):

        # Store input
        self._domain = domain
        self._M_i = M_i
        self._M_e = M_e
        self._I_s = I_s
        self._I_a = I_a

        # Initialize and update parameters if given
        self.parameters = self.default_parameters()
        if params is not None:
            self.parameters.update(params)

        # Set-up function spaces
        k = self.parameters["polynomial_degree"]
        V = FunctionSpace(self._domain, "CG", k)
        U = FunctionSpace(self._domain, "CG", k)

        use_R = self.parameters["use_avg_u_constraint"]
        if use_R:
            R = FunctionSpace(self._domain, "R", 0)
            self.VUR = MixedFunctionSpace((V, U, R))
        else:
            self.VUR = MixedFunctionSpace((V, U))

        # Set-up solution fields:
        if v_ is None:
            self.v_ = Function(V)
        else:
            info_red("Experimental: v_ shipped from elsewhere.")
            self.v_ = v_
        self.vur = Function(self.VUR)

    def solution_fields(self):
        """
        Return tuple of previous and current solution objects.

        Modifying these will modify the solution objects of the solver
        and thus provides a way for setting initial conditions for
        instance.

        *Returns*
          (previous v, current vur) (:py:class:`tuple` of :py:class:`dolfin.Function`)
        """
        return (self.v_, self.vur)

    def solve(self, interval, dt=None):
        """
        Solve the discretization on a given time interval (t0, t1)
        with a given timestep dt and return generator for a tuple of
        the interval and the current solution.

        *Arguments*
          interval (:py:class:`tuple`)
            The time interval for the solve given by (t0, t1)
          dt (int, optional)
            The timestep for the solve. Defaults to length of interval

        *Returns*
          (timestep, solution_fields) via (:py:class:`genexpr`)

        *Example of usage*::

          # Create generator
          solutions = solver.solve((0.0, 1.0), 0.1)

          # Iterate over generator (computes solutions as you go)
          for (interval, solution_fields) in solutions:
            (t0, t1) = interval
            v_, vur = solution_fields
            # do something with the solutions
        """

        # Initial set-up
        # Solve on entire interval if no interval is given.
        (T0, T) = interval
        if dt is None:
            dt = (T - T0)
        t0 = T0
        t1 = T0 + dt

        # Step through time steps until at end time
        while (True) :
            info("Solving on t = (%g, %g)" % (t0, t1))
            self.step((t0, t1))

            # Yield solutions
            yield (t0, t1), self.solution_fields()

            # Break if this is the last step
            if end_of_time(T, t0, t1, dt):
                break

            # If not: update members and move to next time
            # Subfunction assignment would be good here.
            if isinstance(self.v_, Function):
                self.v_.assign(project(self.vur[0], self.v_.function_space()))
            else:
                info_red("Assuming that v_ is updated elsewhere. Experimental.")
            t0 = t1
            t1 = t0 + dt

    def step(self, interval):
        """
        Solve on the given time interval (t0, t1).

        *Arguments*
          interval (:py:class:`tuple`)
            The time interval (t0, t1) for the step

        *Invariants*
          Assuming that v\_ is in the correct state for t0, gives
          self.vur in correct state at t1.
        """

        # Extract interval and thus time-step
        (t0, t1) = interval
        k_n = Constant(t1 - t0)
        theta = self.parameters["theta"]

        # Extract conductivities
        M_i, M_e = self._M_i, self._M_e

        # Define variational formulation
        use_R = self.parameters["use_avg_u_constraint"]
        if use_R:
             (v, u, l) = TrialFunctions(self.VUR)
             (w, q, lamda) = TestFunctions(self.VUR)
        else:
             (v, u) = TrialFunctions(self.VUR)
             (w, q) = TestFunctions(self.VUR)

        Dt_v = (v - self.v_)/k_n
        v_mid = theta*v + (1.0 - theta)*self.v_

        theta_parabolic = (inner(M_i*grad(v_mid), grad(w))*dx
                           + inner(M_i*grad(u), grad(w))*dx)
        theta_elliptic = (inner(M_i*grad(v_mid), grad(q))*dx
                          + inner((M_i + M_e)*grad(u), grad(q))*dx)
        G = Dt_v*w*dx + theta_parabolic + theta_elliptic

        if use_R:
            G += (lamda*u + l*q)*dx

        # Add applied current as source in elliptic equation if
        # applicable
        I_a = self._I_a
        if I_a:
            t = t0 + theta*(t1 - t0)
            I_a.t = t
            G -= I_a*q*dx

        # Add applied stimulus as source in parabolic equation if
        # applicable
        I_s = self._I_s
        if I_s:
            t = t0 + theta*(t1 - t0)
            I_s.t = t
            G -= I_s*w*dx

        # Define variational problem
        a, L = system(G)
        pde = LinearVariationalProblem(a, L, self.vur)

        # Set-up solver
        solver = LinearVariationalSolver(pde)
        solver.parameters.update(self.parameters["linear_variational_solver"])
        solver.solve()

    @staticmethod
    def default_parameters():
        """Initialize and return a set of default parameters

        *Returns*
          A set of parameters (:py:class:`dolfin.Parameters`)

        To inspect all the default parameters, do::

          info(BasicBidomainSolver.default_parameters(), True)
        """

        params = Parameters("BasicBidomainSolver")
        params.add("theta", 0.5)
        params.add("polynomial_degree", 1)
        params.add("use_avg_u_constraint", True)

        params.add(LinearVariationalSolver.default_parameters())
        return params

class BidomainSolver(BasicBidomainSolver):
    __doc__ = BasicBidomainSolver.__doc__

    def __init__(self, domain, M_i, M_e, I_s=None, I_a=None, v_=None,
                 params=None):

        # Call super-class
        BasicBidomainSolver.__init__(self, domain, M_i, M_e,
                                     I_s=I_s, I_a=I_a, v_=v_,
                                     params=params)

        # Create variational forms
        self._timestep = Constant(self.parameters["default_timestep"])
        (self._lhs, self._rhs) = self.variational_forms(self._timestep)

        # Preassemble left-hand side (will be updated if time-step
        # changes)
        debug("Preassembling bidomain matrix (and initializing vector)")
        self._lhs_matrix = assemble(self._lhs, annotate=True)
        self._rhs_vector = Vector(self._lhs_matrix.size(0))

        # Create linear solver (based on parameter choices)
        self._linear_solver, self._update_solver = self._create_linear_solver()

    @property
    def linear_solver(self):
        """The linear solver (:py:class:`dolfin.LUSolver` or
        :py:class:`dolfin.KrylovSolver`)."""
        return self._linear_solver

    def _create_linear_solver(self):
        "Helper function for creating linear solver based on parameters."
        solver_type = self.parameters["linear_solver_type"]

        if solver_type == "direct":
            solver = LUSolver(self._lhs_matrix)
            solver.parameters.update(self.parameters["lu_solver"])
            update_routine = self._update_lu_solver

        elif solver_type == "iterative":

            solver = KrylovSolver("gmres", "amg")
            solver.parameters.update(self.parameters["krylov_solver"])
            solver.set_operator(self._lhs_matrix)

            # Set null space: v = 0, u = constant
            debug("Setting null space")
            null_space = Vector(self.vur.vector())
            self.VUR.sub(1).dofmap().set(null_space, 1.0)
            solver.set_nullspace([null_space])

            update_routine = self._update_krylov_solver
        else:
            error("Unknown linear_solver_type given: %s" % solver_type)

        return (solver, update_routine)

    @staticmethod
    def default_parameters():
        """Initialize and return a set of default parameters

        *Returns*
          A set of parameters (:py:class:`dolfin.Parameters`)

        To inspect all the default parameters, do::

          info(BidomainSolver.default_parameters(), True)
        """

        params = Parameters("BidomainSolver")
        params.add("theta", 0.5)
        params.add("polynomial_degree", 1)
        params.add("default_timestep", 1.0)

        # Set default solver type to be direct
        params.add("linear_solver_type", "direct")
        params.add("use_avg_u_constraint", True)

        # Add default parameters from both LU and Krylov solvers
        params.add(LUSolver.default_parameters())
        params.add(KrylovSolver.default_parameters())

        # Customize default parameters for LUSolver
        params["lu_solver"]["same_nonzero_pattern"] = True

        # Customize default parameters for KrylovSolver
        params["krylov_solver"]["preconditioner"]["same_nonzero_pattern"] = True

        return params

    def variational_forms(self, k_n):
        """Create the variational forms corresponding to the given
        discretization of the given system of equations.

        *Arguments*
          k_n (:py:class:`ufl.Expr` or float)
            The time step

        *Returns*
          (lhs, rhs) (:py:class:`tuple` of :py:class:`ufl.Form`)
        """

        # Extract theta parameter
        theta = self.parameters["theta"]

        # Define variational formulation
        use_R = self.parameters["use_avg_u_constraint"]
        if use_R:
             (v, u, l) = TrialFunctions(self.VUR)
             (w, q, lamda) = TestFunctions(self.VUR)
        else:
             (v, u) = TrialFunctions(self.VUR)
             (w, q) = TestFunctions(self.VUR)

        # Set-up variational problem
        Dt_v_k_n = (v - self.v_)
        v_mid = theta*v + (1.0 - theta)*self.v_
        theta_parabolic = (inner(self._M_i*grad(v_mid), grad(w))*dx
                           + inner(self._M_i*grad(u), grad(w))*dx)
        theta_elliptic = (inner(self._M_i*grad(v_mid), grad(q))*dx
                          + inner((self._M_i + self._M_e)*grad(u), grad(q))*dx)

        G = (Dt_v_k_n*w*dx + k_n*theta_parabolic + k_n*theta_elliptic)

        if use_R:
            G += k_n*(lamda*u + l*q)*dx

        # Add applied current as source in elliptic equation if
        # applicable
        if self._I_a:
            G -= k_n*self._I_a*q*dx

        # Add applied stimulus as source in parabolic equation if
        # applicable
        if self._I_s:
            G -= k_n*self._I_s*w*dx

        (a, L) = system(G)
        return (a, L)

    def step(self, interval):
        """
        Solve on the given time step (t0, t1).

        *Arguments*
          interval (:py:class:`tuple`)
            The time interval (t0, t1) for the step

        *Invariants*
          Assuming that v\_ is in the correct state for t0, gives
          self.vur in correct state at t1.
        """

        # Extract interval and thus time-step
        (t0, t1) = interval
        dt = t1 - t0
        theta = self.parameters["theta"]
        annotate = True
        t = t0 + theta*dt

        # Update matrix and other variables as needed
        timestep_unchanged = (abs(dt - float(self._timestep)) < 1.e-12)
        if not timestep_unchanged:
            self._update_matrix(dt)
        self._update_solver(timestep_unchanged)

        # Update applied current and stimulus (if applicable)
        if self._I_a:
            self._I_a.t = t
        if self._I_s:
            self._I_s.t = t

        # Assemble right-hand-side
        assemble(self._rhs, tensor=self._rhs_vector, annotate=annotate)

        # Solve problem
        self.linear_solver.solve(self.vur.vector(), self._rhs_vector,
                                 annotate=annotate)

    def _update_matrix(self, dt):
        "Helper function for updating the matrix when timestep has changed."

        debug("Timestep has changed, updating matrix and stored timestep")
        annotate = True

        # Update stored timestep
        # FIXME: dolfin_adjoint still can't annotate constant assignment.
        self._timestep.assign(Constant(dt))#, annotate=annotate)

        # Reassemble matrix
        assemble(self._lhs, tensor=self._lhs_matrix, annotate=annotate)

        # Update solvers
        self.linear_solver.set_operator(self._lhs_matrix)

    def _update_lu_solver(self, timestep_unchanged):
        """Helper function for updating an LUSolver depending on
        whether timestep has changed."""

        # Update reuse of factorization parameter in accordance with
        # changes in timestep
        if timestep_unchanged:
            debug("Timestep is unchanged, reusing LU factorization")
            self.linear_solver.parameters["reuse_factorization"] = True
        else:
            debug("Timestep has changed, updating LU factorization")
            self.linear_solver.parameters["reuse_factorization"] = False

    def _update_krylov_solver(self, timestep_unchanged):
        """Helper function for updating a KrylovSolver depending on
        whether timestep has changed."""

        # Update reuse of preconditioner parameter in accordance with
        # changes in timestep
        if timestep_unchanged:
            debug("Timestep is unchanged, reusing preconditioner")
            self.linear_solver.parameters["preconditioner"]["reuse"] = True
        else:
            debug("Timestep has changed, updating preconditioner")
            self.linear_solver.parameters["preconditioner"]["reuse"] = False

        # Set nonzero initial guess if it indeed is nonzero
        if (self.vur.vector().norm("l2") > 1.e-12):
            debug("Initial guess is non-zero.")
            self.linear_solver.parameters["nonzero_initial_guess"] = True
