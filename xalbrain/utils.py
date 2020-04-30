"""This module provides various utilities for internal use."""


__author__ = "Marie E. Rognes (meg@simula.no), 2012--2013"
__all__ = ["state_space", "convergence_rate", "Projecter", "split_function", "time_stepper"]


import math

import dolfin as df

import typing as tp


def import_extension_modules():
    try:
        import extension_modules
    except ModuleNotFoundError as e:
        extension_modules = None
    return extension_modules


def set_ffc_parameters():
    df.parameters["form_compiler"]["cpp_optimize"] = True
    flags = ["-O3", "-ffast-math", "-march=native"]
    df.parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)
    df.parameters["form_compiler"]["quadrature_degree"] = 3


def split_function(vs: df.Function, dim: int) -> tp.Tuple[df.Function, df.Function]:
    """Split a function into the first component and the rest."""
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
        num_states: int,
        family: str = "CG",
        degree: int = 1
) -> df.FunctionSpace:
    """Return function space for the state variables.

    Arguments:
        domain: The mesh.
        num_states: The number of states
        family: The finite element family, defaults to "CG" if None is given.
        degree: Finite element degree.
    """
    if num_states > 1:
        S = df.VectorFunctionSpace(domain, family, degree, num_states)
    else:
        S = df.FunctionSpace(domain, family, degree)
    return S


def time_stepper(t0: float, t1: float, dt: float = None) -> tp.Iterator[tp.Tuple[float, float]]:
    """Generate time intervals between `t0` and `t1` with length `dt`."""
    if t0 >= t1:
        raise ValueError("dt greater than time interval")
    elif dt is None:
        dt = t1 - t0

    _t0 = t0
    _t1 = t0 + dt

    while _t0 < t1:
        yield _t0, _t1
        _t0 += dt
        _t1 += dt


def convergence_rate(
    mesh_size_list: tp.Sequence[float],
    error_list: tp.Sequence[float]
) -> tp.List[float]:
    """Compute and return rates of convergence :math:`r_i` such that

    .. math::

      errors = C hs^r
    """
    msg = "mesh_size_list and error_list must have same length."
    assert (len(mesh_size_list) == len(error_list)), msg
    ln = math.log
    rates = [
        (ln(error_list[i + 1]/error_list[i]))/(ln(mesh_size_list[i + 1]/mesh_size_list[i]))
            for i in range(len(error_list) - 1)
    ]
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

    def __init__(self, V, parameters=None):
        # Set parameters
        self.parameters = self.default_parameters()
        if parameters is not None:
            self.parameters.update(parameters)

        # Set-up mass matrix for L^2 projection
        self.V = V
        self.u = df.TrialFunction(self.V)
        self.v = df.TestFunction(self.V)
        self.m = df.inner(self.u, self.v)*df.dx()
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
        L = df.inner(f, self.v)*df.dx()
        df.assemble(L, tensor=self.b)
        self.solver.solve(u.vector(), self.b)
