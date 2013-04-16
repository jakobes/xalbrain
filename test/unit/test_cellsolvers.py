"""
Unit tests for various types of solvers for cardiac cell models.
"""

__author__ = "Marie E. Rognes (meg@simula.no), 2013"
__all__ = [""]

import unittest
from dolfin import *
from beatadjoint import *

class TestBasicSingleBasicSingleCellSolver(unittest.TestCase):
    "Test functionality for the basic single cell solver."

    def _run_solve(self, model, theta=None):
        dt = 0.01
        interval = (0.0, 2*dt)

        # Initialize solver
        solver = BasicSingleCellSolver(model)
        if theta is not None:
            info("Updating theta to %g" % theta)
            solver.parameters["theta"] = theta

        # Assign initial conditions
        (vs_, vs) = solver.solution_fields()
        vs_.assign(model.initial_conditions())

        # Solve for a couple of steps
        solutions = solver.solve(interval, dt)
        for ((t0, t1), vs) in solutions:
            pass

        # Check that we are at the end time
        self.assertAlmostEqual(t1, 2*dt)
        return vs.vector()

    def _run_step(self, model, theta=None):
        dt = 0.01
        # Initialize solver
        solver = BasicSingleCellSolver(model)
        if theta is not None:
            info("Updating theta to %g" % theta)
            solver.parameters["theta"] = theta

        # Assign initial conditions
        (vs_, vs) = solver.solution_fields()
        vs.assign(model.initial_conditions())

        # Solve for a couple of steps
        vs = solver.step((0.0, dt), vs)
        vs = solver.step((dt, 2*dt), vs)

        return vs.vector()

    def _compare_solve_step(self, model, theta=None):
        model.stimulus = Expression("1000*t", t=0.0)
        vec_solve = self._run_solve(model)
        vec_step = self._run_step(model)
        for i in range(len(vec_solve)):
            self.assertAlmostEqual(vec_solve[i], vec_step[i])

    def test_default_basic_single_cell_solver(self):
        "Test basic single cell solver."
        for Model in supported_cell_models:
            model = Model()
            self._compare_solve_step(model)

    def test_default_basic_single_cell_solver_be(self):
        "Test basic single cell solver with Backward Euler."
        for Model in supported_cell_models:
            model = Model()
            self._compare_solve_step(model, theta=1.0)

    def test_default_basic_single_cell_solver_fe(self):
        "Test basic single cell solver with Forward Euler."
        for Model in supported_cell_models:
            model = Model()
            self._compare_solve_step(model, theta=0.0)

if __name__ == "__main__":
    print("")
    print("Testing cell solvers")
    print("--------------------")
    unittest.main()
