"""This module provides various utilities for internal use."""


__author__ = "Marie E. Rognes (meg@simula.no), 2012--2013"
__all__ = ["state_space", "convergence_rate", "Projecter", "split_function", "time_stepper"]


import math

import dolfin as df

from typing import (
    Tuple,
    List,
    Sequence,
)


def split_function(vs, dim):
    if vs.function_space().ufl_element().num_sub_elements()==dim:
        v = vs[0]
        if dim == 2:
            s = vs[1]
        else:
            s = df.as_vector([vs[i] for i in range(1, dim)])
    else:
        v, s = df.split(vs)
    return v, s


def state_space(
        domain: df.Mesh,
        d: int,
        family: str = None,
        k: int = 1
) -> df.FunctionSpace:
    """
    Return function space for the state variables.

    *Arguments*
      domain (:py:class:`df.Mesh`)
        The computational domain
      d (int)
        The number of states
      family (string, optional)
        The finite element family, defaults to "CG" if None is given.
      k (int, optional)
        The finite element degree, defaults to 1

    *Returns*
      a function space (:py:class:`df.FunctionSpace`)
    """
    if family is None:
        family = "CG"
    if d > 1:
        S = df.VectorFunctionSpace(domain, family, k, d)
    else:
        S = df.FunctionSpace(domain, family, k)
    return S


def time_stepper(t0: float, t1: float, dt: float):
    _t0 = t0
    _t1 = t0 + dt

    while _t0 < t1:
        yield _t0, _t1
        _t0 += dt
        _t1 += dt


def convergence_rate(hs: Sequence[float], errors: Sequence[float]) -> List[float]:
    """
    Compute and return rates of convergence :math:`r_i` such that

    .. math::

      errors = C hs^r
    """
    assert (len(hs) == len(errors)), "hs and errors must have same length."
    ln = math.log
    rates = [(ln(errors[i + 1]/errors[i]))/(ln(hs[i + 1]/hs[i])) for i in range(len(hs) - 1)]
    return rates


class Projecter:
    """Customized class for repeated projection.

    *Arguments*
      V (:py:class:`df.FunctionSpace`)
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
        self.u = df.TrialFunction(self.V)
        self.v = df.TestFunction(self.V)
        self.m = df.inner(self.u, self.v)*dolfin.dx()
        self.M = df.assemble(self.m)
        self.b = df.Vector(V.mesh().mpi_comm(), V.dim())

        solver_type = self.parameters["linear_solver_type"]
        assert(solver_type == "lu" or solver_type == "cg"),  \
            "Expecting 'linear_solver_type' to be 'lu' or 'cg'"
        if solver_type == "lu":
            df.debug("Setting up direct solver for projecter")
            # Customize LU solver (reuse everything)
            solver = df.LUSolver(self.M)
            solver.parameters["same_nonzero_pattern"] = True
            solver.parameters["reuse_factorization"] = True
        else:
            df.debug("Setting up iterative solver for projecter")
            # Customize Krylov solver (reuse everything)
            solver = df.KrylovSolver("cg", "ilu")
            solver.set_operator(self.M)
            solver.parameters["preconditioner"]["structure"] = "same"
            # solver.parameters["nonzero_initial_guess"] = True
        self.solver = solver

    @staticmethod
    def default_parameters():
        parameters = df.Parameters("Projecter")
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
          u (:py:class:`df.Function`)
            The result of the projection
        """
        L = df.inner(f, self.v)*dolfin.dx()
        df.assemble(L, tensor=self.b)
        self.solver.solve(u.vector(), self.b)
