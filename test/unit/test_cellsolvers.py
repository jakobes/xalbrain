"""
Unit tests for various types of solvers for cardiac cell models.
"""

__author__ = "Marie E. Rognes (meg@simula.no), 2013"
__all__ = ["TestBasicSingleCellSolver",
           "TestPointIntegralSolver"]

import unittest
from dolfin import *
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
        dt = 0.01
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
        solutions = solver.solve(interval, dt)
        for ((t0, t1), vs) in solutions:
            pass

        # Check that we are at the end time
        self.assertAlmostEqual(t1, 2*dt)
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
        info("Testing %s" % str(model))

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

    def _test_point_integral_solver(self):
        mesh = UnitIntervalMesh(1)
        for Model in supported_cell_models:
            for Scheme in [BackwardEuler, ForwardEuler, CrankNicolson,
                           RK4, ESDIRK3, ESDIRK4]:
                self._compare_against_reference(Model, Scheme, mesh)


class TestBasicSingleCellSolverAdjoint(unittest.TestCase):
    "Test adjoint functionality for the basic single cell solver."

    def _run(self, theta, model, ics):
        # Initialize solver
        params = BasicSingleCellSolver.default_parameters()
        params["theta"] = theta
        solver = BasicSingleCellSolver(model, params=params)

        # Assign initial conditions
        (vs_, vs) = solver.solution_fields()
        vs_.assign(ics, annotate=True)

        # Solve for a couple of steps
        dt = 0.01
        solutions = solver.solve((0.0, 2.0*dt), dt)
        for ((t0, t1), vs) in solutions:
            pass
        return vs

    def test_replay(self):
        "Test that replay reports success for basic single cell solver"
        # Initialize cell model
        for theta in (1.0, 0.1, 0.5):
            for Model in (FitzHughNagumoManual, Tentusscher_2004_mcell):
                adj_reset()
                model = Model()
                ics = model.initial_conditions()
                info_green("Running %s with %g" % (model, theta))
                vs = self._run(0.5, model, ics)

                info_green("Replaying")
                success = replay_dolfin(tol=0.0, stop=True)
                self.assertEqual(success, True)

if __name__ == "__main__":
    print("")
    print("Testing cell solvers")
    print("--------------------")
    unittest.main()
