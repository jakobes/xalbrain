"""This module provides various utilities for internal use."""

__author__ = "Marie E. Rognes (meg@simula.no), 2012--2013"

__all__ = ["state_space", "end_of_time", "convergence_rate",
           "Projecter"]

import math
import dolfin
import dolfin_adjoint

def state_space(domain, d, family=None, k=1):
    """Return function space for the state variables.

    *Arguments*
      domain (:py:class:`dolfin.Mesh`)
        The computational domain
      d (int)
        The number of states
      family (string, optional)
        The finite element family, defaults to "CG" if None is given.
      k (int, optional)
        The finite element degree, defaults to 1

    *Returns*
      a function space (:py:class:`dolfin.FunctionSpace`)
    """
    if family is None:
        family = "CG"
    if d > 1:
        S = dolfin.VectorFunctionSpace(domain, family, k, d)
    else:
        S = dolfin.FunctionSpace(domain, family, k)
    return S

def end_of_time(T, t0, t1, dt):
    """
    Return True if the interval (t0, t1) is the last before the end
    time T, otherwise False.
    """
    return (t1 + dt) > (T + dolfin.DOLFIN_EPS)

def convergence_rate(hs, errors):
    """
    Compute and return rates of convergence :math:`r_i` such that

    .. math::

      errors = C hs^r

    """
    assert (len(hs) == len(errors)), "hs and errors must have same length."
    # Compute converence rates
    rates = [(math.log(errors[i+1]/errors[i]))/(math.log(hs[i+1]/hs[i]))
             for i in range(len(hs)-1)]

    # Return convergence rates
    return rates

class Projecter(object):
    """Customized class for repeated projection.

    *Arguments*
      V (:py:class:`dolfin.FunctionSpace`)
        The function space to project into
      solver_type (string, optional)
        "iterative" (default) or "direct"

    *Example of usage*::
      my_project = Projecter(V, solver_type="direct")
      u = Function(V)
      f = Function(W)
      my_project(f, u)
    """

    def __init__(self, V, params=None):
        # Set parameters
        self.parameters = self.default_parameters()
        if params is not None:
            self.parameters.update(params)

        # Set-up mass matrix for L^2 projection
        self.V = V
        self.u = dolfin.TrialFunction(self.V)
        self.v = dolfin.TestFunction(self.V)
        self.m = dolfin.inner(self.u, self.v)*dolfin.dx
        self.M = dolfin_adjoint.assemble(self.m)
        self.b = dolfin.Vector(V.dim())

        solver_type = self.parameters["linear_solver_type"]
        assert(solver_type == "lu" or solver_type == "cg"),  \
            "Expecting 'linear_solver_type' to be 'lu' or 'cg'"
        if solver_type == "lu":
            dolfin.debug("Setting up direct solver for projecter")
            # Customize LU solver (reuse everything)
            solver = dolfin.LUSolver(self.M)
            solver.parameters["same_nonzero_pattern"] = True
            solver.parameters["reuse_factorization"] = True
        else:
            dolfin.debug("Setting up iterative solver for projecter")
            # Customize Krylov solver (reuse everything)
            solver = dolfin.KrylovSolver("cg", "ilu")
            solver.set_operator(self.M)
            solver.parameters["preconditioner"]["reuse"] = True
            solver.parameters["preconditioner"]["same_nonzero_pattern"] = True
            # solver.parameters["nonzero_initial_guess"] = True
        self.solver = solver

    @staticmethod
    def default_parameters():
        parameters = dolfin.Parameters("Projecter")
        parameters.add("linear_solver_type", "cg")
        return parameters

    def __call__(self, f, u):
        """
        Carry out projection of ufl Expression f and store result in
        the function u. The user must make sure that u lives in the
        right space.

        *Arguments*
          f (:py:class:`ufl.Expr`)
            The thing to be projected into this function space
          u (:py:class:`dolfin.Function`)
            The result of the projection
        """
        L = dolfin.inner(f, self.v)*dolfin.dx
        dolfin.assemble(L, tensor=self.b)
        self.solver.solve(u.vector(), self.b)

