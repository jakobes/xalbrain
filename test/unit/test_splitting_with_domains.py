"""Unit tests for various types of bidomain solver."""

__author__ = "Marie E. Rognes (meg@simula.no), 2013"
__all__ = ["TestSplittingSolver"]


from xalbrain import (
    Model,
    FitzHughNagumoManual,
    SplittingSolver,
)

import dolfin as df


def test_solver_with_domains() -> None:
    mesh = df.UnitCubeMesh(5, 5, 5)
    time = df.Constant(0.0)

    stimulus = df.Expression("2.0*t", t=time, degree=1)

    # Create ac
    applied_current = df.Expression("sin(2*pi*x[0])*t", t=time, degree=3)

    # Create conductivity "tensors"
    M_i = 1.0
    M_e = 2.0

    cell_model = FitzHughNagumoManual()
    cardiac_model = Model(
        mesh,
        time,
        M_i,
        M_e,
        cell_model,stimulus,
        applied_current
    )

    dt = 0.1
    t0 = 0.0
    dt = dt
    T = t0 + 5*dt

    ics = cell_model.initial_conditions()

    # Create basic solver
    parameters = SplittingSolver.default_parameters()
    parameters["ode_solver_choice"] = "BasicCardiacODESolver"
    solver = SplittingSolver(cardiac_model, parameters=parameters)

    vs_, vs, vur = solver.solution_fields()
    vs_.assign(ics)

    # Solve
    solutions = solver.solve(t0, T, dt)
    for (interval, fields) in solutions:
        (vs_, vs, vur) = fields

if __name__ == "__main__":
    test_solver_with_domains()
