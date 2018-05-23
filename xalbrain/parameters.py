"""Collection of specifications for all the solvers."""
from typing import NamedTuple


class BidomainParameters(NamedTuple):
    """Parameters for Bidomain solver."""
    linear_solver_type: str
    solver: str = "BidomainSolver"        # Not happy with this
    theta: float = 0.5
    polynomial_degree: int = 1
    enable_adjoint: bool = False


class MonodomainParameters(NamedTuple):
    # TODO: Merge with Bidomain Parameters
    linear_solver_type: str
    solver: str = "MonodomainSolver"
    theta: float = 0.5
    enable_adjoint: bool = False
    polynomial_degree: int = 1


class SingleCellParameters(NamedTuple):
    """Parameters for CellSolver."""
    scheme: str     # Both multistagescheme and linear solver

    theta: float = 0.5
    solver: str = "CardiacODESolver"

    V_polynomial_degree: int = 1
    V_polynomial_family: str = "CG"
    S_polynomial_degree: int = 1
    S_polynomial_family: str = "CG"


class SplittingParameters(NamedTuple):
    """Parameters for SplittingSolver."""
    theta: float = 0.5
    apply_stimulus_current_to_pde: bool = False


class KrylovParameters(NamedTuple):
    """Parameters for Krylov solver."""
    solver: str = "gmres"
    preconditioner: str = "petsc_amg"
    absolute_tolerance: float = 1e-14
    relative_tolerance: float = 1e-14
    nonzero_initial_guess: bool = True


class LUParameters(NamedTuple):
    """Parameters for LU solver."""
    solver: str = "default"
    same_nonzero_pattern: bool = True
    reuse_factorization: bool = True
