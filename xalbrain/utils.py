import math

import dolfin as df
import numpy as np

from typing import (
    Union,
    NamedTuple,
    List,
    Sequence,
    Iterator,
    Tuple,
    Any
)


class MonodomainParameters(NamedTuple):
    timestep: df.Constant = df.Constant(1.0)
    theta: df.Constant = df.Constant(0.5)
    linear_solver_type: str = "direct"
    lu_type: str = "default"
    krylov_method: str = "cg"
    krylov_preconditioner: str = "petsc_amg"


class BidomainParameters(NamedTuple):
    restrict_tags: Any = set()
    timestep: df.Constant = df.Constant(1.0)
    theta: df.Constant = df.Constant(0.5)
    linear_solver_type: str = "direct"
    lu_type: str = "default"
    krylov_method: str = "gmres"   # CG fails
    krylov_preconditioner: str = "petsc_amg"


class SplittingSolverParameters(NamedTuple):
    theta: df.Constant = df.Constant(0.5)


class ODESolverParameters(NamedTuple):
    valid_cell_tags: Sequence[int]
    timestep: df.Constant = df.Constant(1)
    reload_extension_modules: bool = False
    theta: df.Constant = df.Constant(0.5)


def load_xdmf_mesh(directory: str, name: str) -> Tuple[df.Mesh, df.MeshFunction, df.MeshFunction]:
    """
    Return a mesh stored as xdmf and the corresponding cell function and facet function if appliccable.
    """
    mesh = df.Mesh()
    with df.XDMFFile(f"{directory}/{name}.xdmf") as infile:
        infile.read(mesh)

    mvc = df.MeshValueCollection("size_t", mesh, 2)
    with df.XDMFFile(f"{directory}/{name}_cf.xdmf") as infile:
        infile.read(mvc, "cell_data")
    cell_function = df.MeshFunction("size_t", mesh, mvc)

    try:
        mvc = df.MeshValueCollection("size_t", mesh, 1)
        with df.XDMFFile(f"{directory}/{name}_ff.xdmf") as infile:
            infile.read(mvc, "facet_data")
        interface_function = df.MeshFunction("size_t", mesh, mvc)
    except RuntimeError:
        interface_function = df.MeshFunction("size_t", mesh, mesh.geometric_dimension() - 1)
        interface_function.set_all(0)
    return mesh, cell_function, interface_function


def create_linear_solver(
        lhs_matrix,
        parameters: CoupledMonodomainParameters
) -> Union[df.LUSolver, df.KrylovSolver]:
    """helper function for creating linear solver."""
    solver_type = parameters.linear_solver_type       # direct or iterative

    if solver_type == "direct":
        solver = df.LUSolver(lhs_matrix, parameters.lu_type)
        solver.parameters["symmetric"] = True

    elif solver_type == "iterative":
        method = parameters.krylov_method
        preconditioner = parameters.krylov_preconditioner

        solver = df.PETScKrylovSolver(method, preconditioner)
        solver.set_operator(lhs_matrix)
        solver.parameters["nonzero_initial_guess"] = True
        solver.ksp().setFromOptions()       # TODO: What is this?
    else:
        raise ValueError(f"Unknown linear_solver_type given: {solver_type}")

    return solver


def time_stepper(*, t0: float, t1: float, dt: float = None) -> Iterator[Tuple[float, float]]:
    """Convenience function to handle time stepping."""
    if dt is None:
        dt = t1 - t0
    elif dt > t1 - t0:
        raise ValueError("dt greater than time interval")

    _t0 = t0
    _t = t0 + dt
    while _t < t1:
        yield _t0, _t
        _t += dt
        _t0 += dt


def state_space(
    domain: df.Mesh,
    num_states: int,
    family: str=None,
    element_degree: int=1
) -> df.FunctionSpace:
    """Return function space for the state variables.

    Arguments:
       domain: The computational domain
       num_states: The number of states
       family: The finite element famile. Defaults to CG.
       element_degree: The finite element degree. Defaults to 1.
    """
    if family is None:
        family = "CG"
    if num_states > 1:
        S = df.VectorFunctionSpace(domain, family, element_degree, num_states)
    else:
        S = df.FunctionSpace(domain, family, element_degree)
    return S


def convergence_rate(hs, errors) -> List[float]:
    """
    Compute and return rates of convergence :math:`r_i` such that

    .. math::

      errors = C hs^r
    """
    if not len(hs) == len(errors):
        raise RuntimeError("hs and errors must have same length")
    rates = [
        (math.log(errors[i + 1]/errors[i]))/(math.log(hs[i + 1]/hs[i])) for i in range(len(hs) - 1)
    ]
    return rates


def splat(vs, dim):
    """Split subspaces."""
    if vs.function_space().ufl_element().num_sub_elements() == dim:
        v = vs[0]
        if dim == 2:
            s = vs[1]
        else:
            s = df.as_vector([vs[i] for i in range(1, dim)])
    else:
        v, s = df.split(vs)
    return v, s
