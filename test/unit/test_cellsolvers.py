"""
Unit tests for various types of solvers for cardiac cell models.
"""
__author__ = "Marie E. Rognes (meg@simula.no), 2013"
__all__ = ["TestCardiacODESolver", "TestBasicSingleCellSolver"]


import itertools
from testutils import slow, assert_almost_equal, parametrize

from dolfin import info, info_green, \
        UnitIntervalMesh, MPI, mpi_comm_world
from beatadjoint import supported_cell_models, \
        CardiacODESolver, BasicSingleCellSolver, \
        NoCellModel, FitzHughNagumoManual, Fitzhughnagumo, Tentusscher_2004_mcell, \
        Constant, Expression


supported_schemes = ["ForwardEuler", "BackwardEuler", "CrankNicolson", "RK4", "ESDIRK3", "ESDIRK4"]

class TestCardiacODESolver(object):
    """ Tests the cardiac ODE solver on different cell models. """

    # Note that these should be essentially identical to the ones
    # for the BasicSingleCellSolver
    references = {NoCellModel:
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

    def compare_against_reference(self, sol, Model, Scheme):
        ''' Compare the model solution with the reference solution. '''
        try:
            ind, ref_value = self.references[Model][Scheme]
        except KeyError:
            info("Missing references for %s, %s: value is %g"
                 % (Model, Scheme, sol[0]))
            return

        info("Value for %s, %s is %g" % (Model, Scheme, sol[ind]))
        if ref_value != "nan":
            assert_almost_equal(float(sol[ind]), float(ref_value), tolerance=1e-6)

    def replace_with_constants(self, params):
        ''' Replace all float values in params by Constants. '''
        for param_name in params.keys():
            value = params[param_name]
            params[param_name] = Constant(value)

    def _setup_solver(self, Model, Scheme, time=0.0, stim=None, params=None):
        ''' Generate a new solver object with the given start time, stimulus and parameters. '''
        # Create model instance
        model = Model(params=params)

        # Initialize time and stimulus (note t=time construction!)
        if stim is None:
            stim = {0: Expression("1000*t", t=time)}

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

    @slow
    @parametrize(("Model","Scheme"), 
        list(itertools.product(supported_cell_models,supported_schemes))
        )
    def test_compare_against_reference(self, Model, Scheme):
        ''' Runs the given cell model with the numerical scheme 
            and compares the result with the reference value. '''

        solver = self._setup_solver(Model, Scheme, time=Constant(0))
        (vs_, vs) = solver.solution_fields()

        next_dt = 0.01
        solver.step((0.0, next_dt))
        vs_.assign(vs)
        solver.step((next_dt, 2*next_dt))

        self.compare_against_reference(vs.vector(), Model, Scheme)

    @slow
    @parametrize(("Model","Scheme"), 
        list(itertools.product(supported_cell_models,supported_schemes))
        )
    def test_compare_against_reference_constant(self, Model, Scheme):
        ''' Runs the given cell model with the numerical scheme 
            and compares the result with the reference value. '''

        params = Model.default_parameters()
        self.replace_with_constants(params)

        solver = self._setup_solver(Model, Scheme, time=Constant(0), params=params)
        (vs_, vs) = solver.solution_fields()

        next_dt = 0.01
        solver.step((0.0, next_dt))
        vs_.assign(vs)
        solver.step((next_dt, 2*next_dt))

        self.compare_against_reference(vs.vector(), Model, Scheme)


class TestBasicSingleCellSolver(object):
    "Test functionality for the basic single cell solver."

    references = {NoCellModel: {1.0: (0, 0.3),
                                 None: (0, 0.2),
                                 0.0: (0, 0.1)},
                   FitzHughNagumoManual: {1.0:  (0, -84.70013280019053),
                                          None: (0, -84.8000503072239979),
                                          0.0:  (0, -84.9)},
                   Tentusscher_2004_mcell: {1.0: (1, -85.89745525156506),
                                            None: (1, -85.99686000794499),
                                            0.0:  (1, -86.09643254164848),}
                   }

    def _run_solve(self, model, time, theta=None):
        "Run two time steps for the given model with the given theta solver."
        dt = 0.01
        T = 2*dt
        interval = (0.0, T)

        # Initialize solver
        params = BasicSingleCellSolver.default_parameters()
        if theta is not None:
            params["theta"] = theta

        params["enable_adjoint"] = False
        solver = BasicSingleCellSolver(model, time, params=params)

        # Assign initial conditions
        (vs_, vs) = solver.solution_fields()
        vs_.assign(model.initial_conditions())

        # Solve for a couple of steps
        solutions = solver.solve(interval, dt)
        for ((t0, t1), vs) in solutions:
            pass

        # Check that we are at the end time
        assert_almost_equal(t1, T, 1e-10)
        return vs.vector()

    def _compare_solve_step(self, Model, theta=None):
        "Set-up model and compare result to precomputed reference if available."
        model = Model()
        time = Constant(0.0)
        model.stimulus = {0:Expression("1000*t", t=time)}
        info_green("\nTesting %s" % model)
        vec_solve = self._run_solve(model, time, theta)
        if Model in self.references and theta in self.references[Model]:
            ind, ref_value = self.references[Model][theta]
            assert_almost_equal(vec_solve[ind], ref_value, 1e-10)
            
        else:
            info("Missing references for %r, %r" % (Model, theta))

    @slow
    def test_default_basic_single_cell_solver(self):
        "Test basic single cell solver."
        for Model in supported_cell_models:
            self._compare_solve_step(Model)

    @slow
    def test_default_basic_single_cell_solver_be(self):
        "Test basic single cell solver with Backward Euler."
        for Model in supported_cell_models:
            self._compare_solve_step(Model, theta=1.0)

    @slow
    def test_default_basic_single_cell_solver_fe(self):
        "Test basic single cell solver with Forward Euler."
        for Model in supported_cell_models:
            self._compare_solve_step(Model, theta=0.0)

