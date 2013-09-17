"""
Unit tests for various types of solvers for cardiac cell models.
"""
from __future__ import division

__author__ = "Marie E. Rognes (meg@simula.no), 2013"
__all__ = ["TestBasicSingleCellSolver",
           "TestCardiacODESolver"]

import unittest
import numpy as np
from dolfin import *
from beatadjoint import *
from beatadjoint.utils import state_space

class TestBasicSingleCellSolver(unittest.TestCase):
    "Test functionality for the basic single cell solver."

    def setUp(self):
        "Set-up references when existing."
        self.references = {NoCellModel: {1.0: (0, 0.3),
                                         None: (0, 0.2),
                                         0.0: (0, 0.1)},
                           FitzHughNagumoManual: {1.0:  (0, -84.70013280019053),
                                                  None: (0, -84.8000503072239979),
                                                  0.0:  (0, -84.9)},
                           Tentusscher_2004_mcell: {1.0: (15, -85.89745525156506),
                                                    None: (15, -85.99686000794499),
                                                    0.0:  (15, -86.09643254164848),}
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
        self.assertAlmostEqual(t1, T)
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
            self.assertAlmostEqual(vec_solve[ind], ref_value)
        else:
            info("Missing references for %r, %r" % (Model, theta))

    def test_default_basic_single_cell_solver(self):
        "Test basic single cell solver."
        if MPI.num_processes() > 1:
            return
        for Model in supported_cell_models:
            self._compare_solve_step(Model)

    def test_default_basic_single_cell_solver_be(self):
        "Test basic single cell solver with Backward Euler."
        if MPI.num_processes() > 1:
            return
        for Model in supported_cell_models:
            self._compare_solve_step(Model, theta=1.0)

    def test_default_basic_single_cell_solver_fe(self):
        "Test basic single cell solver with Forward Euler."
        if MPI.num_processes() > 1:
            return
        for Model in supported_cell_models:
            self._compare_solve_step(Model, theta=0.0)

class TestCardiacODESolver(unittest.TestCase):
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
                           {"BackwardEuler": (15, -85.89745525156506),
                            "CrankNicolson": (15, -85.99685674414921),
                            "ForwardEuler":  (15, -86.09643254164848),
                            "RK4":  (15, "nan"),
                            "ESDIRK3":  (15, -85.99681862337053),
                            "ESDIRK4":  (15, -85.99681796046603),
                            }
                           }

    def _setup_solver(self, Model, Scheme, mesh, time, stim=None, params=None):
        # Create model instance
        model = Model(params=params)

        # Initialize time and stimulus (note t=time construction!)
        if stim is None:
            stim = {0:Expression("1000*t", t=time)}

        # Initialize solver
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

    def _compare_against_reference(self, Model, Scheme, mesh):

        info_blue("Comparing against reference")
        time = Constant(0.0)
        solver = self._setup_solver(Model, Scheme, mesh, time)

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
            info_red("Missing references for %s, %s: value is %g"
                     % (Model, Scheme, vs.vector()[0]))

        # Use Constant Parameters
        params = Model.default_parameters()
        if params:
            for param_name in params.keys():
                value = params[param_name]
                params[param_name] = Constant(value)

            time.assign(0.0)
            solver = self._setup_solver(Model, Scheme, mesh, time, params=params)

            (vs_, vs) = solver.solution_fields()
            solver.step((0.0, next_dt))
            vs_.assign(vs)
            solver.step((next_dt, 2*next_dt))

            vs = solver._scheme.solution()

            if Model in self.references and Scheme in self.references[Model]:
                ind, ref_value = self.references[Model][Scheme]
                info("Value for %s, %s is %g"
                     % (Model, Scheme, vs.vector()[ind]))
                if ref_value != "nan":
                    self.assertAlmostEqual(vs.vector()[ind], ref_value, 6)
            else:
                info("Missing references for %s, %s: value is %g"
                     % (Model, Scheme, vs.vector()[0]))

    def test_cardiac_ode_solver(self):
        "Testing cardiac ode solvers for many models and solvers"
        if MPI.num_processes() > 1:
            return
        mesh = UnitIntervalMesh(1)
        for Model in supported_cell_models:
            for Scheme in ["ForwardEuler", "BackwardEuler", "CrankNicolson",
                           "RK4", "ESDIRK3", "ESDIRK4"]:
                self._compare_against_reference(Model, Scheme, mesh)

    def _long_run_compare(self, mesh, Model, Scheme, dt_org, abs_tol, rel_tol):

        tstop = 10
        ind_V = 0
        dt_ref = 0.1
        time_ref = np.linspace(0, tstop, int(tstop/dt_ref)+1)
        Vm_reference = np.fromfile("Vm_reference.npy")
        params = Model.default_parameters()

        time = Constant(0.0)
        stim = {0:Expression("(time >= stim_start) && (time < stim_start + stim_duration)"\
                             " ? stim_amplitude : 0.0 ", time=time, stim_amplitude=52.0, \
                             stim_start=1.0, stim_duration=1.0)}

        # Initiate solver, with model and Scheme
        if dolfin_adjoint:
            adj_reset()

        solver = self._setup_solver(Model, Scheme, mesh, time, stim, params)
        solver._pi_solver.parameters["newton_solver"]["absolute_tolerance"] = 1e-6
        solver._pi_solver.parameters["newton_solver"]["report"] = False

        scheme = solver._scheme
        (vs_, vs) = solver.solution_fields()

        vs.assign(vs_)

        dof_to_vertex_map_values = dof_to_vertex_map(vs.function_space())
        scheme.t().assign(0.0)

        vs_array = np.zeros(mesh.num_vertices()*\
                            vs.function_space().dofmap().num_entity_dofs(0))
        vs_array[dof_to_vertex_map_values] = vs.vector().array()
        output = [vs_array[ind_V]]
        time_output = [0.0]
        dt = dt_org

        # Time step
        next_dt = max(min(tstop-float(scheme.t()), dt), 0.0)
        t0 = 0.0

        while next_dt > 0.0:

            # Step solver
            solver.step((t0, t0 + next_dt))
            vs_.assign(vs)

            # Collect plt output data
            vs_array[dof_to_vertex_map_values] = vs.vector().array()
            output.append(vs_array[ind_V])
            time_output.append(float(scheme.t()))

            # Next time step
            t0 += next_dt
            next_dt = max(min(tstop-float(scheme.t()), dt), 0.0)

        # Compare solution from CellML run using opencell
        self.assertAlmostEqual(output[-1], Vm_reference[-1], abs_tol)

        output = np.array(output)
        time_output = np.array(time_output)
        output = np.interp(time_ref, time_output, output)

        value = np.sqrt(np.sum(((Vm_reference-output)/Vm_reference)**2))/len(Vm_reference)
        self.assertAlmostEqual(value, 0.0, rel_tol)

    def test_long_run_tentusscher(self):
        mesh = UnitIntervalMesh(5)
        for Scheme, dt_org, abs_tol, rel_tol in [("BackwardEuler", 0.1, 1, 1),
                                                 ("CrankNicolson", 0.1, 0, 1),
                                                 ("ESDIRK3", 0.1, 0, 1),
                                                 ("ESDIRK4", 0.1, 0, 1)]:

            self._long_run_compare(mesh, Tentusscher_2004_mcell, Scheme, \
                                   dt_org, abs_tol, rel_tol)

if __name__ == "__main__":
    print("")
    print("Testing cell solvers")
    print("--------------------")
    unittest.main()
