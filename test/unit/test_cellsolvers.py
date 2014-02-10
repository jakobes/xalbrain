"""
Unit tests for various types of solvers for cardiac cell models.
"""
__author__ = "Marie E. Rognes (meg@simula.no), 2013"

import unittest
from dolfin import *
from beatadjoint import *


class ParametrizedCardiacODESolver(unittest.TestCase):

    def __init__(self, methodName='runTest', Model=None, Scheme=None):

        super(ParametrizedCardiacODESolver, self).__init__(methodName)
        self.Model = Model
        self.Scheme = Scheme

    @staticmethod
    def parametrize(testcase_klass, Model, Scheme):
        """ Create a suite containing all tests taken from the given
            subclass, passing them the parameters
        """
        testloader = unittest.TestLoader()
        testnames = testloader.getTestCaseNames(testcase_klass)
        suite = unittest.TestSuite()
        for name in testnames:
            suite.addTest(testcase_klass(name, Model=Model, Scheme=Scheme))
        return suite


class TestCardiacODESolver(ParametrizedCardiacODESolver):

    def setUp(self):
        # Note that these should be essentially identical to the ones
        # for the BasicSingleCellSolver
        self.references = {NoCellModel:
                           {"BackwardEuler": (0, 0.3),
                            "CrankNicolson": (0, 0.2),
                            "ForwardEuler": (0, 0.1),
                            "RK4": (0, 0.2),
                            "ESDIRK3": (0, 0.2),
                            "ESDIRK4": (0, 0.2),
                            },
                           
                           FitzHughNagumoManual:
                           {"BackwardEuler": (0, -84.70013280019053),
                            "CrankNicolson": (0, -84.80005016079546),
                            "ForwardEuler": (0, -84.9),
                            "RK4": (0, -84.80004467770296),
                            "ESDIRK3": (0, -84.80004459269247),
                            "ESDIRK4": (0, -84.80004468281632),
                            },
                           
                           Fitzhughnagumo:
                           {"BackwardEuler": (0, -84.70013280019053),
                            "CrankNicolson": (0, -84.8000501607955),
                            "ForwardEuler":  (0, -84.9),
                            "RK4":  (0, -84.80004467770296),
                            "ESDIRK3":  (0, -84.80004467770296),
                            "ESDIRK4":  (0, -84.80004468281632),
                            },

                           Tentusscher_2004_mcell:
                           {"BackwardEuler": (1, -85.89745525156506),
                            "CrankNicolson": (1, -85.99685674414921),
                            "ForwardEuler":  (1, -86.09643254164848),
                            "RK4":  (1, "nan"),
                            "ESDIRK3":  (1, -85.99681862337053),
                            "ESDIRK4":  (1, -85.99681796046603),
                            }
                           }

    def _setup_solver(self, Model, Scheme, time, stim=None, params=None):
        # Create model instance
        model = Model(params=params)

        # Initialize time and stimulus (note t=time construction!)
        if stim is None:
            stim = {0:Expression("1000*t", t=time)}

        # Initialize solver
        mesh = UnitIntervalMesh(5)
        params = CardiacODESolver.default_parameters()
        params["scheme"] = Scheme
        solver = CardiacODESolver(mesh, time, model.num_states(),
                                  model.F, model.I, I_s=stim, params=params)

        # Create scheme
        info_green("\nTesting %s with %s scheme" % (model, Scheme))

        # Start with native initial conditions
        (vs_, vs) = solver.solution_fields()
        vs_.assign(model.initial_conditions())
        vs.assign(vs_)

        return solver

    @unittest.skipIf(MPI.size(mpi_comm_world()) > 1, "parallel not supported yet")
    def test_compare_against_reference(self):
        Model = self.Model
        Scheme = self.Scheme

        time = Constant(0.0)
        solver = self._setup_solver(Model, Scheme, time)

        (vs_, vs) = solver.solution_fields()

        next_dt = 0.01
        solver.step((0.0, next_dt))
        vs_.assign(vs)
        solver.step((next_dt, 2*next_dt))

        if Model in self.references and Scheme in self.references[Model]:
            ind, ref_value = self.references[Model][Scheme]
            info("Value for %s, %s is %g"
                 % (Model, Scheme, vs.vector()[ind]))
            if ref_value != "nan":
                self.assertAlmostEqual(vs.vector()[ind], ref_value, 6)
        else:
            self.skipTest("Missing references for %s, %s: value is %g"
                         % (Model, Scheme, vs.vector()[0]))

    @unittest.skipIf(MPI.size(mpi_comm_world()) > 1, "parallel not supported yet")
    def test_compare_against_reference_constant(self):
        Model = self.Model
        Scheme = self.Scheme

        time = Constant(0.0)
        next_dt = 0.01

        # Use Constant Parameters
        params = Model.default_parameters()
        if params:
            for param_name in params.keys():
                value = params[param_name]
                params[param_name] = Constant(value)

        solver = self._setup_solver(Model, Scheme, time, params=params)

        (vs_, vs) = solver.solution_fields()
        solver.step((0.0, next_dt))
        vs_.assign(vs)
        solver.step((next_dt, 2*next_dt))

        if Model in self.references and Scheme in self.references[Model]:
            ind, ref_value = self.references[Model][Scheme]
            info("Value for %s, %s is %g"
                 % (Model, Scheme, vs.vector()[ind]))
            if ref_value != "nan":
                self.assertAlmostEqual(vs.vector()[ind], ref_value, 6)
        else:
            info("Missing references for %s, %s: value is %g"
                 % (Model, Scheme, vs.vector()[0]))


def suite():            

    suite = unittest.TestSuite()
    for Model in supported_cell_models:
        for Scheme in ["ForwardEuler", "BackwardEuler", "CrankNicolson",
                       "RK4", "ESDIRK3", "ESDIRK4"]:

            suite.addTest(ParametrizedCardiacODESolver.parametrize(TestCardiacODESolver, Model=Model, Scheme=Scheme))

    return suite
    unittest.TextTestRunner().run(suite)

        

if __name__ == "__main__":
    print("")
    print("Testing cell solvers")
    print("--------------------")
    unittest.TextTestRunner(verbosity=2).run(suite())
