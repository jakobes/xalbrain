"""
Unit tests for various types of solvers for cardiac cell models.
"""

__author__ = "Marie E. Rognes (meg@simula.no), 2013"
__all__ = ["TestBasicSingleCellSolver",
           "TestPointIntegralSolver"]

import unittest
from dolfin import *
from dolfin_adjoint import *
from beatadjoint import *
from beatadjoint.utils import state_space

class TestBasicSingleCellSolver(unittest.TestCase):
    "Test functionality for the basic single cell solver."

    def setUp(self):
        "Set-up references when existing."
        self.references = {NoCellModel: {1.0: 0.3, None: 0.2, 0.0: 0.1},
                           FitzHughNagumoManual: {1.0:  -84.70013280019053,
                                                  None: -84.80005016079546,
                                                  0.0:  -84.9}}

    def _run_solve(self, model, theta=None):
        "Run two time steps for the given model with the given theta solver."
        dt = 0.001
        interval = (0.0, 2*dt)

        # Initialize solver
        params = BasicSingleCellSolver.default_parameters()
        if theta is not None:
            params["theta"] = theta
        solver = BasicSingleCellSolver(model, params)

        # Assign initial conditions
        (vs_, vs) = solver.solution_fields()
        vs_.assign(model.initial_conditions())

        # Solve for a couple of steps
        T = 2*dt
        solutions = solver.solve(interval, T)
        for ((t0, t1), vs) in solutions:
            pass

        # Check that we are at the end time
        self.assertAlmostEqual(t1, T)
        return vs.vector()

class TestPointIntegralSolver(unittest.TestCase):
    def setUp(self):
        # Note that these should be (and are) identical to the ones
        # for the BasicSingleCellSolver
        self.references = {NoCellModel:
                           {BackwardEuler: (0, 0.3),
                            CrankNicolson: (0, 0.2),
                            ForwardEuler: (0, 0.1),
                            RK4 : (0, 0.2),
                            ESDIRK3 : (0, 0.2),
                            ESDIRK4 : (0, 0.2),
                            },

                           FitzHughNagumoManual:
                           {BackwardEuler: (0, -84.70013280019053),
                            CrankNicolson: (0, -84.80005016079546),
                            ForwardEuler:  (0, -84.9),
                            RK4:  (0, -84.80004467770296),
                            ESDIRK3:  (0, -84.80004468383603),
                            ESDIRK4:  (0, -84.80004468281632),
                            },

                           Fitzhughnagumo:
                           {BackwardEuler: (0, -84.69986709136005),
                            CrankNicolson: (0, -84.79994981706433),
                            ForwardEuler:  (0, -84.9),
                            RK4:  (0, -84.79995530744164),
                            ESDIRK3:  (0, -84.79995530333677),
                            ESDIRK4:  (0, -84.79995530333677),
                            },

                           Tentusscher_2004_mcell:
                           {BackwardEuler: (15, -85.8974552517),
                            CrankNicolson: (15, -85.99685674422098),
                            ForwardEuler:  (15, -86.09643254167213),
                            RK4:  (15, "nan"),
                            ESDIRK3:  (15, -85.9968179615449),
                            ESDIRK4:  (15, -85.9968179605287),
                            }
                           }

    def _compare_against_reference(self, Model, Scheme, mesh):

        # Create model instance
        model = Model()
        info_green("Testing %s" % str(model))

        # Initialize time and stimulus (note t=time construction!)
        time = Constant(0.0)
        model.stimulus = Expression("1000*t", t=time)

        # Create rhs form by combining cell model info with function space
        V = FunctionSpace(mesh, "CG", 1)
        S = state_space(mesh, model.num_states())
        VS = V*S
        vs = Function(VS)
        (v, s) = split(vs)
        (w, q) = TestFunctions(VS)
        rhs = (inner(model.F(v, s), q) - inner(model.I(v, s), w))*dP
        if model.stimulus:
            rhs += inner(model.stimulus, w)*dP

        # Create scheme
        # NOTE: No need to initialize time to 0. That is done in the
        # NOTE: constructor of the Scheme
        scheme = Scheme(rhs, vs, time)

        # Start with native initial conditions, step twice and compare
        # results to given reference
        next_dt = 0.01
        vs.assign(model.initial_conditions())
        solver = PointIntegralSolver(scheme)
        solver.parameters.newton_solver.report = False
        solver.step(next_dt)
        solver.step(next_dt)

        if Model in self.references and Scheme in self.references[Model]:
            ind, ref_value = self.references[Model][Scheme]
            info("Value for %s, %s is %g"
                 % (Model, Scheme, vs.vector()[ind]))
            if ref_value != "nan":
                self.assertAlmostEqual(vs.vector()[ind], ref_value)
        else:
            info("Missing references for %s, %s: value is %g"
                 % (Model, Scheme, vs.vector()[0]))

    def test_point_integral_solver(self):
        mesh = UnitIntervalMesh(1)
        for Model in supported_cell_models:
            for Scheme in [BackwardEuler, ForwardEuler, CrankNicolson,
                           RK4, ESDIRK3, ESDIRK4]:
                self._compare_against_reference(Model, Scheme, mesh)


class TestBasicSingleCellSolverAdjoint(unittest.TestCase):
    "Test adjoint functionality for the basic single cell solver."

    def _run(self, solver, model, ics):
        # Assign initial conditions
        (vs_, vs) = solver.solution_fields()
        vs_.assign(ics)

        # Solve for a couple of steps
        dt = 0.01
        solutions = solver.solve((0.0, 2*dt), dt)
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
                solver = BasicSingleCellSolver(model, params=params)

                info_green("Running %s with %g" % (model, theta))

                ics = Function(project(model.initial_conditions(), solver.VS),
                               name="ics")
                self._run(solver, model, ics)

                info_green("Replaying")
                success = replay_dolfin(tol=0.0, stop=True)
                self.assertEqual(success, True)

    def test_compute_adjoint(self):
        "Test that we can compute the adjoint for some given functional"

        for theta in (0.5,):# 0.0, 1.0):
            for Model in (FitzHughNagumoManual, Tentusscher_2004_mcell):
                adj_reset()
                model = Model()

                params = BasicSingleCellSolver.default_parameters()
                params["theta"] = theta
                solver = BasicSingleCellSolver(model, params=params)

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

        for theta in (0.5, 0.0, 1.0):
            for Model in (FitzHughNagumoManual, Tentusscher_2004_mcell):
                adj_reset()
                model = Model()

                params = BasicSingleCellSolver.default_parameters()
                params["theta"] = theta
                solver = BasicSingleCellSolver(model, params=params)

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
        "."

        for theta in (0.0, 0.5, 1.0):
            for Model in (FitzHughNagumoManual, ):#Tentusscher_2004_mcell):

                adj_reset()
                model = Model()

                params = BasicSingleCellSolver.default_parameters()
                params["theta"] = theta
                solver = BasicSingleCellSolver(model, params=params)

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
                conv_rate = taylor_test(Jhat, InitialConditionParameter(vs_),
                                        Jics, dJdics)

                # Check that minimal rate is greater than some given number
                self.assertGreater(conv_rate, 1.98)

if __name__ == "__main__":
    print("")
    print("Testing cell solvers")
    print("--------------------")
    unittest.main()
