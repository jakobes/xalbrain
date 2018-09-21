"""Test that the pde + external Neumann current works."""

import pytest

import numpy as np

from xalbrain import (
    UnitSquareMesh,
    CardiacModel,
    SplittingSolver,
    Constant,
    Expression,
)

from dolfin import File

from xalbrain.cellmodels import Wei 


def test_ect_current():
    """Test that the Neumann BC on the extracellular potential is applied correctly."""
    time = Constant(0)
    T = 1e-1
    dt = 1e-5
    mesh = UnitSquareMesh(75, 75)
    ect_current = Expression(
        "std::abs(sin(2*pi*70*t)) > 0.8 ? 800*(x[0] < 0.5 && x[1] < 0.5)*(t < t0) : 0",
        t=time,
        t0=10,
        degree=1
    )
    model = Wei()
    brain = CardiacModel(mesh, time, 0.1, 0.3, model, ect_current=ect_current)
    params = SplittingSolver.default_parameters()
    
    ps = SplittingSolver.default_parameters()
    ps["pde_solver"] = "bidomain"       # TODO: parametrise mono/bi
    ps["BidomainSolver"]["linear_solver_type"] = "direct"
    ps["theta"] = 0.5       # TODO: parametrise theta
    solver = SplittingSolver(brain, params=ps)

    vs_, vs, _ = solver.solution_fields()
    assert False, "Use the other weimodel initial conditions."
    vs_.assign(model.initial_conditions())

    solutions = solver.solve((0, T), dt)
    outfile = File("ect_testdir/v.pvd")
    counter = 0
    for interval, (vs_, vs, _) in solutions:
        print(counter)
        v, *_ = vs.split()
        outfile << v
        counter += 1


if __name__ == "__main__":
    test_ect_current()
