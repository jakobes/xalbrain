"""Collection of specifications for all the solvers."""
from typing import NamedTuple


class BidomainParameters(NamedTuple):
    """Parameters for Bidomain solver."""
    linear_solver_type: str
    theta: float = 0.5
    polynomial_degree: int = 1
    enable_adjoint: bool = False

    # TODO: Make dependent on choise of solver. Do this in solver.__init__
    use_avg_u_constraint: bool      # Use True if direct solver


class MonodomainParameters(NamedTuple):
    # TODO: Merge with Bidomain Parameters
    linear_solver_type: str = "direct"
    theta: float = 0.5
    enable_adjoint: bool = False
    polynomial_degree: int = 1


class SingleCellParameters(NamedTuple):
    """Parameters for CellSolver."""
    scheme: str     # Both multistagescheme and linear solver

    theta: float = 0.5

    V_polynomial_degree: int = 0
    V_polynomial_family: str = "DG"
    S_polynomial_degree: int = 0
    S_polynomial_family: str = "DG"


class SplittingParameters(NamedTuple):
    """Parameters for SplittingSolver."""
    pde_solver: str = "bidomain"
    ode_solver: str = "cellsolver"
    theta: float = 0.5
    apply_current_to_pde: bool = False


class KrylovParmeters(NamedTuple):
    """Parameters for Krylov solver."""
    solver: str = "gmres"
    preconditioner: str = "petsc_amg"
    absolute_tolerance: float = 1e-14
    relative_tolerance: float = 1e-14
    nonzero_initial_gues: bool = True


class LUParameters(NamedTuple):
    """Parameters for LU solver."""
    solver: str = "default"
    lu_same_nonzero_pattern: bool = True
