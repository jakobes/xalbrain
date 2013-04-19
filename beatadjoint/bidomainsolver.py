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
from beatadjoint.utils import end_of_time

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
    def __init__(self, domain, M_i, M_e, I_s=None, I_a=None, params=None):

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
            #info_blue("Solving on t = (%g, %g)" % (t0, t1))
            self._step((t0, t1))

            # Yield solutions
            yield (t0, t1), self.solution_fields()

            # Break if this is the last step
            if end_of_time(T, t0, t1, dt):
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

        Dt_v = (v - self.v_)/k_n
        v_mid = theta*v + (1.0 - theta)*self.v_

        theta_parabolic = (inner(M_i*grad(v_mid), grad(w))*dx
                           + inner(M_i*grad(u), grad(w))*dx)
        theta_elliptic = (inner(M_i*grad(v_mid), grad(q))*dx
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

    # Create domain
    level = 3
    N = 10*(2**level)
    mesh = UnitSquareMesh(N, N)

    # Create stimulus
    ac_str = "cos(t)*cos(2*pi*x[0])*cos(2*pi*x[1]) + 4*pow(pi, 2)*cos(2*pi*x[0])*cos(2*pi*x[1])*sin(t)"
    stimulus = Expression(ac_str, t=0, degree=5)

    # Create conductivity "tensors"
    M_i = 1.0
    M_e = 1.0

    # Set-up solver
    solver = BidomainSolver(mesh, M_i, M_e, I_s=stimulus)
    theta = solver.parameters["theta"]

    # Define end-time and (constant) timestep
    dt = 0.001#/(2**level)
    T = 20*dt

    # Define exact solution (Note: v is returned at end of time
    # interval(s), u is computed at somewhere in the time interval
    # depending on theta)
    v_exact = Expression("cos(2*pi*x[0])*cos(2*pi*x[1])*sin(t)", t=T, degree=3)
    u_exact = Expression("-cos(2*pi*x[0])*cos(2*pi*x[1])*sin(t)/2.0",
                         t=T - (1. - theta)*dt, degree=3)

    # Define initial condition(s)
    (v_, vur) = solver.solution_fields()

    # Solve
    solutions = solver.solve((0, T), dt)
    info_green("Solving primal")
    for (interval, fields) in solutions:
        #print "interval = ", interval
        #plot(stimulus, title="Stimulus", mesh=mesh)
        continue
    interactive()
    (v, u, r) = vur.split(deepcopy=True)

    # Compute errors
    v_error = errornorm(v_exact, v, "L2", degree_rise=2)
    u_error = errornorm(u_exact, u, "L2", degree_rise=2)
    print "v_error = %.16e" % v_error
    print "u_error = %.16e" % u_error

"""
Convergence tests:

v_error = 5.1913975566026706e-04
u_error = 2.5095794862192537e-04

v_error = 1.2709198977645071e-04
u_error = 6.1389756255939929e-05

v_error = 3.1607671581472865e-05
u_error = 1.5264892592415211e-05

v_error = 7.8928214705244280e-06
u_error = 3.8119244660922680e-06

"""
