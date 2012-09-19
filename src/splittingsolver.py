# Copyright (C) 2012 Marie E. Rognes (meg@simula.no)
# Use and modify at will
# Last changed: 2012-09-19

__all__ = ["SplittingSolver"]

from dolfin import *
try:
    from dolfin_adjoint import *
except:
    print "dolfin_adjoint not found. Install it or mod this solver"
    exit()

import utils

class SplittingSolver:
    """Operator splitting based solver for the bidomain equations.

    The splitting algorithm can be controlled by the parameter
    'theta'.  theta = 1.0 corresponds to a (1st order) Godunov
    splitting, theta = 0.5 to a (2nd order) Strang splitting.

    See p. 78 ff in Sundnes et al 2006 for details.
    """
    def __init__(self, model, parameters=None):
        "Create solver."

        # Set model and parameters
        self._model = model
        self._parameters = self.default_parameters()
        if parameters is not None:
            self._parameters.update(parameters)

        # Extract solution domain
        domain = self._model.domain()

        # Create function spaces
        k = self._parameters["potential_polynomial_degree"]
        l = self._parameters["ode_polynomial_degree"]
        num_states = self._model.cell_model().num_states()

        self.V = FunctionSpace(domain, "CG", k)
        R = FunctionSpace(domain, "R", 0)
        self.VUR = MixedFunctionSpace([self.V, self.V, R])
        if num_states > 1:
            self.S = VectorFunctionSpace(domain, "DG", l, num_states)
        else:
            self.S = FunctionSpace(domain, "DG", l)
        self.VS = self.V*self.S

        # Helper functions
        self.u = Function(self.VUR.sub(1).collapse())
        self.vs_ = Function(self.VS)
        self.vs = Function(self.VS)

    def default_parameters(self):

        parameters = Parameters("SplittingSolver")
        parameters.add("enable_adjoint", False)
        parameters.add("theta", 0.5)
        parameters.add("potential_polynomial_degree", 1)
        parameters.add("ode_polynomial_degree", 0)

        return parameters

    def solution_fields(self):
        return (self.vs_, self.vs, self.u)

    def solve(self, interval, dt):

        # Initial set-up
        (T0, T) = interval
        t0 = T0; t1 = T0 + dt
        vs0 = self.vs_

        while (t1 <= T):
            # Solve
            info_blue("Solving on t = (%g, %g)" % (t0, t1))
            timestep = (t0, t1)
            self.step(timestep, vs0)

            # Update
            t0 = t1; t1 = t0 + dt

    def step(self, interval, ics):
        "Step through given interval with given initial conditions"

        # Extract some parameters for readability
        theta = self._parameters["theta"]
        annotate = self._parameters["enable_adjoint"]

        # Extract time domain
        (t0, t1) = interval
        dt = (t1 - t0)
        t = t0 + theta*dt

        # Compute tentative membrane potential and state (vs_star)
        vs_star = self.ode_step((t0, t), ics)
        (v_star, s_star) = split(vs_star)

        # Compute tentative potentials vu = (v, u)
        vur = self.pde_step((t0, t1), v_star)
        (v, u, r) = split(vur)

        # Merge (inverse of split) v and s_star:
        v_s_star = utils.join((v, s_star), self.VS, annotate=annotate)

        # If first order splitting, we are essentially done:
        if theta == 1.0:
            self.vs.assign(v_s_star, annotate=annotate)
        # Otherwise, we do another ode_step:
        else:
            vs = self.ode_step((t, t1), v_s_star)
            self.vs.assign(vs, annotate=annotate)

        # Update previous
        self.vs_.assign(self.vs)

        # Store u (Not a part of the solution algorithm, no need to
        # annotate, fortunately ..)
        self.u.assign(vur.split()[1], annotate=False)

    def ode_step(self, interval, ics):
        """
        Solve

        v_t = - I_ion(v, s)
        s_t = F(v, s)

        with v(t0) = v_, s(t0) = s_
        """
        # For now, ust use theta scheme. To be improved.

        # Extract time domain
        (t0, t1) = interval
        k_n = Constant(t1 - t0)

        # Extract initial conditions
        (v_, s_) = split(ics)

        # Set-up current variables
        vs = Function(self.VS)
        vs.assign(ics) # Start with good guess
        (v, s) = split(vs)
        (w, r) = TestFunctions(self.VS)

        # Define equation based on cell model
        # Note sign for I_theta
        Dt_v = (v - v_)/k_n
        Dt_s = (s - s_)/k_n

        theta = self._parameters["theta"]
        F = self._model.cell_model().F
        I_ion = self._model.cell_model().I
        I_theta = - (theta*I_ion(v, s) + (1 - theta)*I_ion(v_, s_))
        F_theta = theta*F(v, s) + (1 - theta)*F(v_, s_)

        # Set-up system
        G = (Dt_v - I_theta)*w*dx + inner(Dt_s - F_theta, r)*dx

        # Solve system
        pde = NonlinearVariationalProblem(G, vs, J=derivative(G, vs))
        solver = NonlinearVariationalSolver(pde)
        #solver.parameters["newton_solver"]["relative_tolerance"] = 1.e-16
        #solver.parameters["newton_solver"]["absolute_tolerance"] = 1.e-16
        #solver.parameters["newton_solver"]["maximum_iterations"] = 10
        solver.solve(annotate=self._parameters["enable_adjoint"])

        return vs

    def pde_step(self, interval, v_):
        """
        Solve

        v_t - div(M_i grad(v) ..) = 0
        div (M_i grad(v) + ..) = 0

        with v(t0) = v_,
        """

        # Extract interval and time-step
        (t0, t1) = interval
        k_n = Constant(t1 - t0)
        theta = self._parameters["theta"]
        annotate = self._parameters["enable_adjoint"]

        # Extract conductivities from model
        M_i, M_e = self._model.conductivities()

        # Define variational formulation
        (v, u, r) = TrialFunctions(self.VUR)
        (w, q, s) = TestFunctions(self.VUR)

        Dt_v = (v - v_)/k_n
        theta_parabolic = (theta*inner(M_i*grad(v), grad(w))*dx
                           + (1.0 - theta)*inner(M_i*grad(v_), grad(w))*dx
                           + inner(M_i*grad(u), grad(w))*dx)
        theta_elliptic = (theta*inner(M_i*grad(v), grad(q))*dx
                          + (1.0 - theta)*inner(M_i*grad(v_), grad(q))*dx
                          + inner((M_i + M_e)*grad(u), grad(q))*dx)
        G = (Dt_v*w*dx + theta_parabolic + theta_elliptic
             + (s*u + r*q)*dx)

        if self._model.applied_current:
            t = t0 + theta*(t1 - t0)
            self._model.applied_current.t = t
            G -= self._model.applied_current*w*dx

        a, L = system(G)

        # Solve system
        vur = Function(self.VUR)
        pde = LinearVariationalProblem(a, L, vur)
        solver = LinearVariationalSolver(pde)
        solver.solve(annotate=annotate)
        return vur
