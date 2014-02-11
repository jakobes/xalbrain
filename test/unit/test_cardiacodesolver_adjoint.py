"""
Unit tests for various types of solvers for cardiac cell models.
"""

__author__ = "Marie E. Rognes (meg@simula.no), 2013"
__all__ = ["TestCardiacODESolverAdjoint"]

from testutils import assert_equal, assert_true, assert_greater, slow, adjoint

import types
from beatadjoint.dolfinimport import UnitIntervalMesh, info_green, \
        MPI, mpi_comm_world
from beatadjoint import supported_cell_models, \
        Tentusscher_2004_mcell, \
        CardiacODESolver, \
        adj_reset, replay_dolfin, InitialConditionParameter, \
        Constant, Expression, Function, Functional, \
        project, inner, assemble, dx, dt, FINISH_TIME, \
        parameters, compute_gradient_tlm, compute_gradient, \
        taylor_test

parameters["form_compiler"]["cpp_optimize"] = True
flags = "-O3 -ffast-math -march=native"
parameters["form_compiler"]["cpp_optimize_flags"] = flags

def single_cell_closure(Scheme, Model):

    @adjoint
    @slow
    def test_replay(self):
        mesh = UnitIntervalMesh(1)
        if Model in [Tentusscher_2004_mcell] and Scheme in \
           ["ForwardEuler", "RK4"]:
            return

        # Initiate solver, with model and Scheme
        adj_reset()
        params = Model.default_parameters()
        model = Model(params=params)

        solver = self._setup_solver(model, Scheme, mesh)
        ics = project(model.initial_conditions(), solver.VS)

        info_green("Running forward %s with %s A" % (model, Scheme))
        self._run(solver, ics)

        info_green("Replaying")

        # FIXME: Can we increase the tolerance?
        success = replay_dolfin(tol=0, stop=True)
        assert_true(success)
        
        "Test that we can compute the gradient for some given functional"
        if MPI.size(mpi_comm_world()) > 1:
            return
        mesh = UnitIntervalMesh(1)
        
        #if Model in [Tentusscher_2004_mcell]:
        #    return

        if Model in [Tentusscher_2004_mcell] and Scheme in \
           ["ForwardEuler", "RK4"]:
            return

        adj_reset()

        # Initiate solver, with model and Scheme
        params = Model.default_parameters()
        model = Model(params=params)

        solver = self._setup_solver(model, Scheme, mesh)
        ics = Function(project(model.initial_conditions(), solver.VS), name="ics")

        info_green("Running forward %s with %s B" % (model, Scheme))
        self._run(solver, ics)

        # Define functional
        (vs_, vs) = solver.solution_fields()
        form = lambda w: inner(w, w)*dx
        J = Functional(form(vs)*dt[FINISH_TIME])

        # Compute value of functional with current ics
        Jics = assemble(form(vs))

        # Seed for taylor test
        if isinstance(model, Tentusscher_2004_mcell):
            seed=1.e-5
        else:
            seed=None

        # Set-up runner
        def Jhat(ics):
            self._run(solver, ics)
            (vs_, vs) = solver.solution_fields()
            return assemble(form(vs))

        # Stop annotating
        parameters["adjoint"]["stop_annotating"] = True

        # Compute gradient with respect to vs. 
        info_green("Computing gradient")
        m = InitialConditionParameter(vs)

        # Check TLM correctness
        dJdics = compute_gradient_tlm(J, InitialConditionParameter(vs), forget=False)
        assert (dJdics is not None), "Gradient is None (#fail)."
        conv_rate_tlm = taylor_test(Jhat, InitialConditionParameter(vs), Jics, dJdics, seed=seed)

        assert_greater(conv_rate_tlm, 1.8)

        # Check ADM correctness
        dJdics = compute_gradient(J, InitialConditionParameter(vs), forget=False)
        assert (dJdics is not None), "Gradient is None (#fail)."
        conv_rate = taylor_test(Jhat, InitialConditionParameter(vs), Jics, dJdics, seed=seed)

        # Check that minimal rate is greater than some given number
        assert_greater(conv_rate, 1.8)


    @adjoint
    @slow
    def test_taylor_remainder(self):
        "Test that we can compute the gradient for some given functional"
        if MPI.size(mpi_comm_world()) > 1:
            return
        mesh = UnitIntervalMesh(1)

        #if Model in [Tentusscher_2004_mcell]:
        #    return

        if Model in [Tentusscher_2004_mcell] and Scheme in \
           ["ForwardEuler", "RK4"]:
            return

        adj_reset()

        # Initiate solver, with model and Scheme
        params = Model.default_parameters()
        model = Model(params=params)

        solver = self._setup_solver(model, Scheme, mesh)
        ics = Function(project(model.initial_conditions(), solver.VS), name="ics")

        info_green("Running forward %s with %s C" % (model, Scheme))
        self._run(solver, ics)

        # Define functional
        (vs_, vs) = solver.solution_fields()
        form = lambda w: inner(w, w)*dx
        J = Functional(form(vs)*dt[FINISH_TIME])

        # Compute value of functional with current ics
        Jics = assemble(form(vs))

        # Seed for taylor test
        if isinstance(model, Tentusscher_2004_mcell):
            seed=1.e-5
        else:
            seed=None

        # Set-up runner
        def Jhat(ics):
            self._run(solver, ics)
            (vs_, vs) = solver.solution_fields()
            return assemble(form(vs))

        # Stop annotating
        parameters["adjoint"]["stop_annotating"] = True

        # Compute gradient with respect to vs. 
        info_green("Computing gradient")
        m = InitialConditionParameter(vs)

        # Check TLM correctness
        dJdics = compute_gradient_tlm(J, InitialConditionParameter(vs), forget=False)
        assert (dJdics is not None), "Gradient is None (#fail)."
        conv_rate_tlm = taylor_test(Jhat, InitialConditionParameter(vs), Jics, dJdics, seed=seed)

        assert_greater(conv_rate_tlm, 1.8)

        # Check ADM correctness
        dJdics = compute_gradient(J, InitialConditionParameter(vs), forget=False)
        assert (dJdics is not None), "Gradient is None (#fail)."
        conv_rate = taylor_test(Jhat, InitialConditionParameter(vs), Jics, dJdics, seed=seed)

        # Check that minimal rate is greater than some given number
        assert_greater(conv_rate, 1.8)
    
    # Return functions with Scheme and Model fixed
    return tuple(func for func in locals().values() if isinstance(func, types.FunctionType))

class TestCardiacODESolverAdjoint(object):
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

        return solver

    def _run(self, solver, ics):
        # Assign initial conditions
        
        solver._pi_solver.scheme().t().assign(0)
        (vs_, vs) = solver.solution_fields()
        vs_.assign(ics)
        
        # Solve for a couple of steps
        dt = 0.01
        T = 4*dt
        dt = [(0.0, dt), (dt*3,dt/2)]
        solver._pi_solver.parameters.reset_stage_solutions = True
        solver._pi_solver.parameters.newton_solver.reset_each_step = True
        solver._pi_solver.parameters.newton_solver.absolute_tolerance = 1.0e-10
        solver._pi_solver.parameters.newton_solver.recompute_jacobian_for_linear_problems = True
        solutions = solver.solve((0.0, T), dt)
        for ((t0, t1), vs) in solutions:
            pass


for Model in supported_cell_models:
    for Scheme in ["ForwardEuler", "BackwardEuler",
                   "CrankNicolson","RK4", "ESDIRK3", "ESDIRK4"
                   ]:
        for func in single_cell_closure(Scheme, Model):
            method_name = func.func_name+"_"+Scheme+"_"+Model.__name__
            setattr(TestCardiacODESolverAdjoint, method_name, func)
