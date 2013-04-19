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

__all__ = ["BidomainSolver"]

from dolfin import *
from dolfin_adjoint import *
#from beatadjoint import CardiacModel
#from beatadjoint.utils import join, state_space

class BidomainSolver:
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

      params (:py:class:`dolfin.Parameters`, optional)
        Solver parameters

      """
    def __init__(self, domain, M_i, M_e,
                 I_s=None, I_a=None, params=None):

        # Store input
        self._domain = domain
        self._M_i = M_i
        self._M_e = M_e
        self._I_a = I_a
        self._I_s = I_s

        # Initialize and update parameters if given
        self.parameters = self.default_parameters()
        if params is not None:
            self.parameters.update(params)

        # Set-up function spaces
        V = FunctionSpace(self._domain, "CG", 1)
        U = FunctionSpace(self._domain, "CG", 1)
        R = FunctionSpace(self._domain, "R", 0)
        self.VUR = MixedFunctionSpace((V, U, R))

        # Solution fields:
        self.v_ = Function(V)
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
          (timestep, current vur) via (:py:class:`genexpr`)

        *Example of usage*::

          # Create generator
          solutions = solver.solve((0.0, 1.0), 0.1)

          # Iterate over generator (computes solutions as you go)
          for (interval, vur) in solutions:
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
            #info_blue("Solving on t = (%g, %g)" % (t0, t1))
            self._step((t0, t1))

            # Yield solutions
            yield (t0, t1), self.solution_fields()

            # Break if this is the last step
            if ((t1 + dt) > T):
                break

            # If not: update members and move to next time
            # Subfunction assignment would be good here.
            self.v_.assign(project(self.vur[0], self.v_.function_space()))
            t0 = t1
            t1 = t0 + dt

    def _step(self, interval):
        """
        Solve on the given time step (t0, t1).

        *Arguments*
          interval (:py:class:`tuple`)
            The time interval (t0, t1) for the step
        """

        # Extract interval and thus time-step
        (t0, t1) = interval
        k_n = Constant(t1 - t0)
        theta = self.parameters["theta"]
        #annotate = beat_parameters["enable_adjoint"]

        # Extract conductivities from model
        M_i, M_e = self._M_i, self._M_e

        # Define variational formulation
        (v, u, l) = TrialFunctions(self.VUR)
        (w, q, lamda) = TestFunctions(self.VUR)

        v_ = self.v_
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
        #solver_params = self.parameters["linear_variational_solver"]
        #solver.parameters.update(solver_params)

        # Solve system
        #solver.solve(annotate=annotate)
        solver.solve()

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
        return params


if __name__ == "__main__":

    # Input
    mesh = UnitSquare(10, 10)
    M_i = 1.0
    M_e = 1.0

    # Set-up solver and initial conditions
    solver = BidomainSolver(mesh, M_i, M_e)
    v_, vur = solver.solution_fields()
    v_.vector()[:] = 1.0

    dt = 0.1
    interval = (0.0, 4*dt)

    # Solve
    solutions = solver.solve(interval, dt)
    for (timestep, bar) in solutions:
        print timestep
        plot(vur[0], interactive=True)
        print vur.vector().array()

    interactive()
