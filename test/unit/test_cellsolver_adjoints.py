"""
Unit tests for various types of solvers for cardiac cell models.
"""

__author__ = "Marie E. Rognes (meg@simula.no), 2013"
__all__ = ["TestBasicSingleCellSolverAdjoint"]


import unittest
from dolfin import *
from dolfin_adjoint import *
from beatadjoint import *

# TODO: Check with these parameters!
#parameters["form_compiler"]["cpp_optimize"] = True
#flags = "-O3 -ffast-math -march=native"
#parameters["form_compiler"]["cpp_optimize_flags"] = flags

class TestBasicSingleCellSolverAdjoint(unittest.TestCase):
    "Test adjoint functionality for the basic single cell solver."

    def _run(self, solver, model, ics):
        # Assign initial conditions
        (vs_, vs) = solver.solution_fields()
        vs_.assign(ics)

        # Solve for a couple of steps
        dt = 0.01
        T = 2*dt
        solutions = solver.solve((0.0, T), dt)
        for ((t0, t1), vs) in solutions:
            pass

    def test_replay(self):
        "Test that replay reports success for basic single cell solver"
        # Initialize cell model

        for theta in (1.0, 0.0, 0.5):
            for Model in (FitzHughNagumoManual, Tentusscher_2004_mcell):
                adj_reset()
                model = Model()

                # Initialize solver
                params = BasicSingleCellSolver.default_parameters()
                params["theta"] = theta
                solver = BasicSingleCellSolver(model, None, params=params)

                info_green("Running %s with %g" % (model, theta))

                ics = Function(project(model.initial_conditions(), solver.VS),
                               name="ics")
                self._run(solver, model, ics)

                info_green("Replaying")
                success = replay_dolfin(tol=0.0, stop=True)
                self.assertEqual(success, True)

    def test_compute_adjoint(self):
        "Test that we can compute the adjoint for some given functional"

        for theta in (0.0, 0.5, 1.0):
            for Model in (FitzHughNagumoManual, Tentusscher_2004_mcell):
                adj_reset()
                model = Model()

                params = BasicSingleCellSolver.default_parameters()
                params["theta"] = theta
                solver = BasicSingleCellSolver(model, None, params=params)

                # Get initial conditions (Projection of expressions
                # don't get annotated, which is fine, because there is
                # no need.)
                ics = project(model.initial_conditions(), solver.VS)

                # Run forward model
                info_green("Running forward %s with theta %g" % (model, theta))
                self._run(solver, model, ics)

                (vs_, vs) = solver.solution_fields()

                adj_html("forward.html", "forward")
                adj_html("adjoint.html", "adjoint")

                # Define functional and compute gradient etc
                J = Functional(inner(vs, vs)*dx*dt[FINISH_TIME])

                # Compute adjoint
                info_green("Computing adjoint")
                z = compute_adjoint(J)

                # Check that no vs_ adjoint is None (== 0.0!)
                for (value, var) in z:
                    if var.name == "vs_":
                        msg = "Adjoint solution for _vs is None (#fail)."
                        assert (value is not None), msg

    def test_compute_gradient(self):
        "Test that we can compute the gradient for some given functional"

        for theta in (0.0, 0.5, 1.0):
            for Model in (FitzHughNagumoManual, Tentusscher_2004_mcell):
                adj_reset()
                model = Model()

                params = BasicSingleCellSolver.default_parameters()
                params["theta"] = theta
                solver = BasicSingleCellSolver(model, None, params=params)

                # Get initial conditions (Projection of expressions
                # don't get annotated, which is fine, because there is
                # no need.)
                ics = project(model.initial_conditions(), solver.VS)

                # Run forward model
                info_green("Running forward %s with theta %g" % (model, theta))
                self._run(solver, model, ics)

                adj_html("forward.html", "forward")
                adj_html("adjoint.html", "adjoint")

                # Define functional
                (vs_, vs) = solver.solution_fields()
                J = Functional(inner(vs, vs)*dx*dt[FINISH_TIME])

                # Compute gradient with respect to vs_. Highly unclear
                # why with respect to ics and vs fail.
                info_green("Computing gradient")
                dJdics = compute_gradient(J, InitialConditionParameter(vs_))
                assert (dJdics is not None), "Gradient is None (#fail)."
                print dJdics.vector().array()

    def test_taylor_remainder(self):
        "Run Taylor remainder tests for selection of models and solvers."
        for theta in (0.0, 0.5, 1.0):
            for Model in (FitzHughNagumoManual, Fitzhughnagumo,
                          Tentusscher_2004_mcell):

                adj_reset()
                model = Model()

                params = BasicSingleCellSolver.default_parameters()
                params["theta"] = theta
                solver = BasicSingleCellSolver(model, None, params=params)

                # Get initial conditions (Projection of expressions
                # don't get annotated, which is fine, because there is
                # no need.)
                ics = project(model.initial_conditions(), solver.VS)

                # Run forward model
                info_green("Running forward %s with theta %g" % (model, theta))
                self._run(solver, model, ics)

                adj_html("forward.html", "forward")
                adj_html("adjoint.html", "adjoint")

                # Define functional
                (vs_, vs) = solver.solution_fields()
                form = lambda w: inner(w, w)*dx
                J = Functional(form(vs)*dt[FINISH_TIME])

                # Compute value of functional with current ics
                Jics = assemble(form(vs))

                # Compute gradient with respect to vs_ (ics?)
                dJdics = compute_gradient(J, InitialConditionParameter(vs_),
                                          forget=False)

                # Stop annotating
                parameters["adjoint"]["stop_annotating"] = True

                # Set-up runner
                def Jhat(ics):
                    self._run(solver, model, ics)
                    (vs_, vs) = solver.solution_fields()
                    return assemble(form(vs))

                # Run taylor test
                if isinstance(model, Tentusscher_2004_mcell):
                    seed=1.e-4
                else:
                    seed=None
                conv_rate = taylor_test(Jhat, InitialConditionParameter(vs_),
                                        Jics, dJdics, seed=seed)

                # Check that minimal rate is greater than some given number
                self.assertGreater(conv_rate, 1.8)

if __name__ == "__main__":
    print("")
    print("Testing adjoints of cell solvers")
    print("--------------------------------")
    unittest.main()
