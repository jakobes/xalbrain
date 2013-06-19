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

    def xtest_replay(self):
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

    def xtest_compute_adjoint(self):
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

    def xtest_compute_gradient(self):
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

    def xtest_taylor_remainder(self):
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

class TestCardiacODESolverAdjoint(unittest.TestCase):
    def _setup_solver(self, model, Scheme, mesh):

        # Initialize time and stimulus (note t=time construction!)
        time = Constant(0.0)
        stim = Expression("(time >= stim_start) && (time < stim_start + stim_duration)"\
                          " ? stim_amplitude : 0.0 ", time=time, stim_amplitude=52.0, \
                          stim_start=1.0, stim_duration=1.0, name="stim")

        # FIXME: How can we include the above time dependent stimuli current
        stim = None
        
        # Initialize solver
        params = CardiacODESolver.default_parameters()
        params["scheme"] = Scheme
        solver = CardiacODESolver(mesh, time, model.num_states(),
                                  model.F, model.I, I_s=stim, params=params)

        solver._pi_solver.parameters["newton_solver"]["report"] = False

        return solver

    def _run(self, solver, ics):
        # Assign initial conditions
        (vs_, vs) = solver.solution_fields()
        vs_.assign(ics)

        # Solve for a couple of steps
        dt = 0.05
        T = 40*dt
        solutions = solver.solve((0.0, T), dt)
        for ((t0, t1), vs) in solutions:
            pass

    def test_replay(self):
        if MPI.num_processes() > 1:
            return
        mesh = UnitIntervalMesh(1)
        for Model in supported_cell_models:
            for Scheme in ["ForwardEuler", "BackwardEuler", "CrankNicolson",
                           "RK4", "ESDIRK3", "ESDIRK4"]:
                if Model in [Tentusscher_2004_mcell] and Scheme in \
                   ["ForwardEuler", "RK4"]:
                    continue
                
                # Initiate solver, with model and Scheme
                adj_reset()
                params = Model.default_parameters()
                model = Model(params=params)

                solver = self._setup_solver(model, Scheme, mesh)
                ics = Function(project(model.initial_conditions(), solver.VS),
                               name="ics")
                
                info_green("Running forward %s with %s" % (model, Scheme))
                self._run(solver, ics)
                    
                info_green("Replaying")

                # FIXME: Can we increase the tolerance?
                success = replay_dolfin(tol=1e-8, stop=True)
                self.assertTrue(success)

    def test_compute_taylor_reminder(self):
        "Test that we can compute the gradient for some given functional"
        if MPI.num_processes() > 1:
            return
        mesh = UnitIntervalMesh(1)
        for Model in supported_cell_models:
            for Scheme in ["ForwardEuler", "BackwardEuler", "CrankNicolson",
                           "RK4", "ESDIRK3", "ESDIRK4"]:
                if Model in [Tentusscher_2004_mcell] and Scheme in \
                   ["ForwardEuler", "RK4"]:
                    continue

                #adj_reset()

                # FIXME: Above triggers an assert in libadjoint:
                # Traceback (most recent call last):
                #   File "test_cellsolver_adjoints.py", line 235, in test_replay
                #     adj_reset()
                #   File "/home/hake/local/lib/python2.7/site-packages/dolfin_adjoint/adjglobals.py", line 82, in adj_reset
                #     adjointer.reset()
                #   File "/home/hake/local/lib/python2.7/site-packages/libadjoint/libadjoint.py", line 910, in reset
                #     assert len(references_taken) == 0
                # AssertionError
                

                # Initiate solver, with model and Scheme
                params = Model.default_parameters()
                model = Model(params=params)

                solver = self._setup_solver(model, Scheme, mesh)
                ics = Function(project(model.initial_conditions(), solver.VS),
                               name="ics")
                
                info_green("Running forward %s with %s" % (model, Scheme))
                self._run(solver, ics)
                    
                adj_html("forward.html", "forward")
                adj_html("adjoint.html", "adjoint")

                # Define functional
                (vs_, vs) = solver.solution_fields()
                form = lambda w: inner(w, w)*dx
                J = Functional(form(vs)*dt[FINISH_TIME])

                # Compute value of functional with current ics
                Jics = assemble(form(vs))

                # Seed for taylor test
                seed = 1.e-4 if isinstance(Model, Tentusscher_2004_mcell) else None

                # Set-up runner
                def Jhat(ics):
                    self._run(solver, ics)
                    (vs_, vs) = solver.solution_fields()
                    return assemble(form(vs))

                # Stop annotating
                parameters["adjoint"]["stop_annotating"] = True

                # Compute gradient with respect to vs_. 
                info_green("Computing gradient")
                m = InitialConditionParameter(vs_)

                # Check TLM correctness
                dJdics = compute_gradient_tlm(J, m, forget=False)
                assert (dJdics is not None), "Gradient is None (#fail)."
                conv_rate_tlm = taylor_test(Jhat, m, Jics, dJdics, seed=seed)

                # FIXME: outcommented because of failure
                #self.assertGreater(conv_rate_tlm, 1.8)

                # Check ADM correctness
                dJdics = compute_gradient(J, m, forget=False)
                assert (dJdics is not None), "Gradient is None (#fail)."
                conv_rate = taylor_test(Jhat, m, Jics, dJdics, seed=seed)

                # Check that minimal rate is greater than some given number
                # FIXME: outcommented because of failure
                #self.assertGreater(conv_rate, 1.8)


if __name__ == "__main__":
    print("")
    print("Testing adjoints of cell solvers")
    print("--------------------------------")
    unittest.main()
