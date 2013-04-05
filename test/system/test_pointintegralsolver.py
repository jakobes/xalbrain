"""
"""

__author__ = "Marie E. Rognes (meg@simula.no), 2013"
__all__ = []

from dolfin import *
from beatadjoint import *

def main():
    parameters["form_compiler"]["cpp_optimize"] = True

    # Create cell model
    cell = FitzHughNagumoManual()
    num_states = cell.num_states()

    # Create function spaces
    mesh = UnitSquareMesh(2, 2)
    V = FunctionSpace(mesh, "CG", 1)
    S = BasicSplittingSolver.state_space(mesh, num_states)
    VS = V*S

    # Create solution function and set its initial value
    vs = Function(VS)
    (v, s) = split(vs)

    # Define the right-hand-side of the system of ODEs: Dt(u) = rhs(u)
    # Note that sign of the ionic current
    (w, r) = TestFunctions(VS)
    rhs = inner(cell.F(v, s), r) + inner(- cell.I(v, s), w)
    form = rhs*dP

    # In the beginning...
    time = Constant(0.0)

    # Create scheme
    scheme = BackwardEuler(form, vs, time)
    info(scheme)


if __name__ == "__main__":
    main()
