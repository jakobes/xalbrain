"""
"""

__author__ = "Marie E. Rognes (meg@simula.no), 2013"
__all__ = []

from dolfin import *
from beatadjoint import *

def main(CellModel, Solver):

    parameters["form_compiler"]["quadrature_degree"] = 2
    #parameters["form_compiler"]["cpp_optimize"] = True
    flags = ["-O3", "-ffast-math", "-march=native"]
    parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)

    # Create cell model
    cell = CellModel()
    num_states = cell.num_states()

    # Create function spaces
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 1)
    S = BasicSplittingSolver.state_space(mesh, num_states)
    VS = V*S

    # Create solution function and set its initial value
    vs = Function(VS)
    vs.assign(project(cell.initial_conditions(), VS))
    (v, s) = split(vs)

    # Define the right-hand-side of the system of ODEs: Dt(u) = rhs(u)
    # Note that sign of the ionic current
    (w, r) = TestFunctions(VS)
    rhs = inner(cell.F(v, s), r) + inner(- cell.I(v, s), w)
    form = rhs*dP

    # In the beginning...
    time = Constant(0.0)

    # Create scheme
    scheme = Solver(form, vs, time)
    scheme.t().assign(float(time))  # FIXME: Should this be scheme or
                                    # solver and why is this needed?
    info(scheme)

    # Create solver
    solver = PointIntegralSolver(scheme)

    # Time step
    dt = 0.1
    try:
        solver.step(dt)
    except:
        pass

    list_timings()

if __name__ == "__main__":

    Solver = BackwardEuler
    CellModel = supported_cell_models[2]

    print "Testing %s" % CellModel
    main(CellModel, Solver)
