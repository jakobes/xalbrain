# Copyright (C) 2012 Marie E. Rognes (meg@simula.no)
# Use and modify at will
# Last changed: 2012-05-29

__all__ = ["SplittingSolver"]

from dolfin import *
try:
    from dolfin_adjoint import *
except:
    print "dolfin_adjoint not found. Install it or mod this solver"
    exit()

import utils

class SplittingSolver:
    """Operator splitting based solver for the bidomain equations."""
    def __init__(self, model, parameters=None):
        "Create solver."

        # Set model and parameters
        self._model = model
        self._parameters = self.default_parameters()
        if parameters is not None:
            self._parameters.update(parameters)

        # Extract theta parameter
        self._theta = self._parameters["theta"]

        # Extract solution domain
        domain = self._model.domain()

        # Create function spaces
        k = self._parameters["potential_polynomial_degree"]
        l = self._parameters["ode_polynomial_degree"]
        num_states = self._model.cell_model().num_states()

        self.VU = VectorFunctionSpace(domain, "CG", k, 2)
        if num_states > 1:
            self.S = VectorFunctionSpace(domain, "DG", l, num_states)
        else:
            self.S = FunctionSpace(domain, "DG", l)
        self.V = self.VU.sub(0).collapse()
        self.VS = self.V*self.S

        # Internal helper functions
        self._u = Function(self.VU.sub(1).collapse())
        self._vs = Function(self.VS)

    def default_parameters(self):

        parameters = Parameters("SplittingSolver")
        parameters.add("enable_adjoint", False)
        parameters.add("theta", 0.5)
        parameters.add("potential_polynomial_degree", 1)
        parameters.add("ode_polynomial_degree", 0)

        return parameters

    def solution_fields(self):
        return (self._vs, self._u)

    def solve(self, interval, dt):

        # Initial set-up
        (T0, T) = interval
        t0 = T0; t1 = T0 + dt
        vs0 = self._vs

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
        theta = self._theta
        annotate = self._parameters["enable_adjoint"]

        # Extract time domain
        (t0, t1) = interval
        dt = (t1 - t0)
        t = t0 + theta*dt

        # Compute tentative membrane potential and state (vs_star)
        #vs_star = self.ode_step((t0, t), ics)
        vs_star = ics # Debugging
        (v_star, s_star) = split(vs_star)

        # Compute tentative potentials vu = (v, u)
        vu = self.pde_step((t0, t1), v_star)
        (v, u) = split(vu)

        # Merge (inverse of split) v and s_star:
        #v_s_star = utils.merge((v, s_star), self.VS,
        #                       annotate=annotate)

        # If first order splitting, we are essentially done:
        if theta == 1.0:
            self._vs.assign(vs_star)
            #self._vs.assign(v_s_star, annotate=annotate)

        # Otherwise, we do another ode_step:
        #else:
        #    vs = self.ode_step((t, t1), v_s_star)
        #    self._vs.assign(vs, annotate=annotate)

        # Store u (Not a part of the solution algorithm, no need to
        # annotate, fortunately ..)
        #self._u.assign(vu.split()[1], annotate=False)

    def ode_step(self, interval, ics):
        """
        Solve

        v_t = - I_ion(v, s)
        s_t = F(v, s)

        with v(t0) = v_, s(t0) = s_
        """
        # For now, just use theta scheme. To be improved.

        # Extract time domain
        (t0, t1) = interval
        k_n = Constant(t1 - t0)

        # Extract initial conditions
        (v_, s_) = split(ics)

        # Set-up current variables
        vs = Function(self.VS, ics)
        (v, s) = split(vs)
        (w, r) = TestFunctions(self.VS)

        # Define equation based on cell model
        Dt_v = (v - v_)/k_n
        Dt_s = (s - s_)/k_n

        theta = self._theta
        F = self._model.cell_model().F
        I_ion = self._model.cell_model().I
        I_theta = theta*I_ion(v, s) + (1 - theta)*I_ion(v_, s_)
        F_theta = theta*F(v, s) + (1 - theta)*F(v_, s_)

        # Set-up system
        G = (Dt_v + I_theta)*w*dx + inner(Dt_s - F_theta, r)*dx

        # Solve system
        pde = NonlinearVariationalProblem(G, vs, J=derivative(G, vs))
        solver = NonlinearVariationalSolver(pde)
        solver.parameters["newton_solver"]["relative_tolerance"] = 1.e-16
        solver.parameters["newton_solver"]["absolute_tolerance"] = 1.e-16
        solver.solve(annotate=self._parameters["enable_adjoint"])

        return vs

    def pde_step(self, interval, ics):
        """
        Solve

        v_t - div(M_i grad(v) ..) = 0
        div (M_i grad(v) + ..) = 0

        with v(t0) = v_,
        """
        (t0, t1) = interval
        k_n = Constant(t1 - t0)

        M_i, M_e = self._model.conductivities()

        v_ = ics

        (v, u) = TrialFunctions(self.VU)
        (w, q) = TestFunctions(self.VU)

        vu = Function(self.VU)
        Dt_v = (v - v_)/k_n

        theta = self._theta
        annotate = self._parameters["enable_adjoint"]

        theta_parabolic = (theta*inner(M_i*grad(v), grad(w))*dx
                           + (1.0 - theta)*inner(M_i*grad(v_), grad(w))*dx
                           + inner(M_i*grad(u), grad(w))*dx)
        theta_elliptic = (theta*inner(M_i*grad(v), grad(q))*dx
                          + (1.0 - theta)*inner(M_i*grad(v_), grad(q))*dx
                          + inner((M_i + M_e)*grad(u), grad(q))*dx)
        G = (Dt_v*w*dx + theta_parabolic + theta_elliptic)
        a, L = system(G)

        # Solve system
        pde = LinearVariationalProblem(a, L, vu, bcs=None)
        solver = LinearVariationalSolver(pde)
        solver.solve(annotate=annotate)

        return vu

# class ODESolver:
#     def __init__(self, parameters=None):
#         self._parameters = self.default_parameters()
#         if parameters is not None:
#             self._parameters.update(parameters)

#     def step(self, interval, rhs, ics):
#         pass

#     def default_parameters(self):
#         parameters = Parameters("ODESolver")
#         return parameters
