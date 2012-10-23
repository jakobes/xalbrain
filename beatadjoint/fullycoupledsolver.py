"""This module contains fully coupled solvers for (subclasses of)
CardiacModel. Mainly for testing/debugging/comparison purposes"""

# Copyright (C) 2012 Marie E. Rognes (meg@simula.no)
# Use and modify at will
# Last changed: 2012-10-23

__all__ = ["CoupledBidomainSolver"]

from dolfin import *
from dolfin_adjoint import *

from beatadjoint import CardiacModel
from beatadjoint.utils import join

class CoupledBidomainSolver:
    """Basic solver for the bidomain equations (no cell-model)"""

    def __init__(self, model, parameters=None):
        "Create solver from given Cardiac Model and (optional) parameters."

        assert isinstance(model, CardiacModel), \
            "Expecting CardiacModel as first argument"

        # FIXME: Check that we have no cell_model

        # Set model and parameters
        self._model = model
        self.parameters = self.default_parameters()
        if parameters is not None:
            self.parameters.update(parameters)

        # Extract solution domain
        domain = self._model.domain()
        self._domain = domain

        # Create function spaces
        k = self.parameters["potential_polynomial_degree"]
        use_r = self.parameters["real_constraint"]
        self.V = FunctionSpace(domain, "CG", k)

        if use_r:
            R = FunctionSpace(domain, "R", 0)
            self.W= MixedFunctionSpace([self.V, self.V, R])
        else:
            self.W= MixedFunctionSpace([self.V, self.V])

        self.w_ = Function(self.W)
        self.w = Function(self.W)

    @staticmethod
    def default_parameters():
        "Set-up and return default parameters."
        parameters = Parameters("CoupledBidomainSolver")
        parameters.add("enable_adjoint", False)
        parameters.add("theta", 0.5)
        parameters.add("real_constraint", True)
        parameters.add("potential_polynomial_degree", 1)

        pde_solver_params = LinearVariationalSolver.default_parameters()
        parameters.add(pde_solver_params)
        return parameters

    def solution_fields(self):
        "Return tuple of: (previous vu(r), current vu(r))"
        return (self.w_, self.w)

    def solve(self, interval, dt):
        """
        Return generator for solutions on given time interval (t0, t1)
        with given timestep 'dt'.
        """
        # Initial set-up
        (T0, T) = interval
        t0 = T0
        t1 = T0 + dt
        annotate = self.parameters["enable_adjoint"]

        # Step through time steps until at end time.
        while (t1 <= T + DOLFIN_EPS):
            info_blue("Solving on t = (%g, %g)" % (t0, t1))
            timestep = (t0, t1)
            w = self.step(timestep, self.w_)
            self.w.assign(w, annotate=annotate)

            # Yield current solutions
            yield (timestep, self.w)

            # Update previous and timetime
            self.w_.assign(self.w, annotate=annotate)
            t0 = t1
            t1 = t0 + dt

    def step(self, interval, ics):
        "Step through given 'interval' with given initial conditions."

        # Extract some parameters for readability
        theta = self.parameters["theta"]
        annotate = self.parameters["enable_adjoint"]
        use_r = self.parameters["real_constraint"]

        # Extract time domain
        (t0, t1) = interval
        dt = (t1 - t0)
        t = t0 + theta*dt
        k_n = Constant(t1 - t0)

        # Hack, not sure if this is a good design (previous value for
        # u should not be required as data)
        v_ = split(ics)[0]

        # Extract conductivities from model
        M_i, M_e = self._model.conductivities()

        # Define variational formulation
        if use_r:
            (v, u, l) = TrialFunctions(self.W)
            (w, q, lamda) = TestFunctions(self.W)
        else:
            (v, u) = TrialFunctions(self.W)
            (w, q) = TestFunctions(self.W)

        Dt_v = (v - v_)/k_n
        theta_parabolic = (theta*inner(M_i*grad(v), grad(w))*dx
                           + (1.0 - theta)*inner(M_i*grad(v_), grad(w))*dx
                           + inner(M_i*grad(u), grad(w))*dx)
        theta_elliptic = (theta*inner(M_i*grad(v), grad(q))*dx
                          + (1.0 - theta)*inner(M_i*grad(v_), grad(q))*dx
                          + inner((M_i + M_e)*grad(u), grad(q))*dx)
        G = (Dt_v*w*dx + theta_parabolic + theta_elliptic)

        if use_r:
            G += (lamda*u + l*q)*dx

        if self._model.stimulus:
            t = t0 + theta*(t1 - t0)
            self._model.stimulus.t = t
            G -= self._model.stimulus*w*dx

        if self._model.applied_current:
            t = t0 + theta*(t1 - t0)
            self._model.applied_current.t = t
            G -= self._model.stimulus*q*dx

        # Define variational problem
        a, L = system(G)
        A = assemble(a)
        w = Function(self.W)
        pde = LinearVariationalProblem(a, L, w)

        # Set-up solver
        solver = LinearVariationalSolver(pde)
        solver_params = self.parameters["linear_variational_solver"]
        solver.parameters.update(solver_params)

        # Solve system
        solver.solve(annotate=annotate)

        # If not using r, normalize u:
        if not use_r:
            normalization_type = "average"
            assert(not self.parameters["enable_adjoint"]),\
                "Annotation not enabled for normalization"

            (v, u) = w.split(deepcopy=True)
            x = u.vector()
            info_blue("Normalizing ...")
            normalize(x, normalization_type)
            w = project(as_vector((v, u)), self.W, annotate=False)

        return w
