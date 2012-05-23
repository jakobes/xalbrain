# Copyright (C) 2012 Marie E. Rognes (meg@simula.no)
# Use and modify at will
# Last changed: 2012-05-23

__all__ = ["SplittingSolver"]

from dolfin import *

try:
    import dolfin_adjoint
except:
    print "dolfin_adjoint not found. Disabling adjoint annotation"

class SplittingSolver:
    """Operator splitting based solver for the bidomain equations."""
    def __init__(self, model, parameters=None):
        "Create solver."

        self._model = model

        # Set parameters
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

        # Create solution fields
        self.v = Function(self.V)
        self.u = Function(self.VU.sub(1).collapse())
        self.s = Function(self.S)

    def default_parameters(self):

        parameters = Parameters("SplittingSolver")
        parameters.add("enable_adjoint", False)
        parameters.add("theta", 0.5)
        parameters.add("potential_polynomial_degree", 1)
        parameters.add("ode_polynomial_degree", 0)

        return parameters

    def solution_fields(self):
        return (self.v, self.u, self.s)

    def solve(self, interval, dt):

        (T0, T) = interval
        (v0, s0) = (self.v, self.s)

        t0 = T0
        t1 = T0 + dt
        while (t1 <= T):
            info_blue("Solving on t = (%g, %g)" % (t0, t1))

            # Solve
            timestep = (t0, t1)
            self.step(timestep, (v0, s0))

            # Update
            t0 = t1
            t1 = t0 + dt

            #plot(self.v, title="v")
            #plot(self.u, title="u")
            #plot(self.s, title="s")

    def step(self, interval, ics):
        "Step through given interval with given initial conditions"

        theta = self._theta

        # Extract time domain
        (t0, t1) = interval
        dt = (t1 - t0)
        t = t0 + theta*dt

        # Compute tentative membrane potential (v_star) and state (s_star)
        (v_star, s_star) = self.ode_step((t0, t), ics)

        # Compute tentative potentials vu = (v, u)
        (v, u) = self.pde_step((t0, t1), v_star)

        # Compute final membrane potential and state (if not done)
        if theta < 1:
            (v, s) = self.ode_step((t, t1), (v, s_star))
        else:
            s = s_star

        # Update solution fields
        self.v.assign(v)
        self.u.assign(u)
        self.s.assign(s)

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
        (v_, s_) = ics
        vs = project(as_vector((v_, s_)), self.VS)
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

        # Solve system here
        G = (Dt_v + I_theta)*w*dx + inner(Dt_s - F_theta, r)*dx

        if self._parameters["enable_adjoint"]:
            solve(G == 0, vs, annotate=True)
        else:
            solve(G == 0, vs)
        return vs.split()

    def pde_step(self, interval, ics):
        """
        Solve

        v_t - div(M_i grad(v) ...) = 0
        div (M_i grad(v) + ...) = 0

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

        theta_parabolic = (theta*inner(M_i*grad(v), grad(w))*dx
                           + (1.0 - theta)*inner(M_i*grad(v_), grad(w))*dx
                           + inner(M_i*grad(u), grad(w))*dx)
        theta_elliptic = (theta*inner(M_i*grad(v), grad(q))*dx
                          + (1.0 - theta)*inner(M_i*grad(v_), grad(q))*dx
                          + inner((M_i + M_e)*grad(u), grad(q))*dx)
        G = (Dt_v*w*dx + theta_parabolic + theta_elliptic)
        a, L = system(G)
        #bcs = DirichletBC(self.VU.sub(1), 0.0, "on_boundary")
        if self._parameters["enable_adjoint"]:
            # , bcs) # Here we can probably optimize away
            solve(a == L, vu, annotate=True)
        else:
            solve(a == L, vu)
        return vu.split()

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
