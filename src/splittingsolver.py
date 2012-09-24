# Copyright (C) 2012 Marie E. Rognes (meg@simula.no)
# Use and modify at will
# Last changed: 2012-09-24

__all__ = ["SplittingSolver", "BasicSplittingSolver"]

from dolfin import *
try:
    from dolfin_adjoint import *
except:
    print "dolfin_adjoint not found. Install it or mod this solver"
    exit()

import utils

class BasicSplittingSolver:
    """Operator splitting based solver for the bidomain equations.

    The splitting algorithm can be controlled by the parameter
    'theta'.  theta = 1.0 corresponds to a (1st order) Godunov
    splitting, theta = 0.5 to a (2nd order) Strang splitting.

    See p. 78 ff in Sundnes et al 2006 for details.

    Assumes that conductivities does not change over time.
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
        self._domain = domain

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

        self._v = Function(self.V)
        self._s = Function(self.S)

    def default_parameters(self):

        parameters = Parameters("BasicSplittingSolver")
        parameters.add("enable_adjoint", False)
        parameters.add("theta", 0.5)
        parameters.add("linear_pde_solver", "direct")

        parameters.add("potential_polynomial_degree", 1)
        parameters.add("ode_polynomial_degree", 0)

        parameters.add("plot_solutions", False)
        parameters.add("store_solutions", False)

        return parameters

    def solution_fields(self):
        return (self.vs_, self.vs, self.u)

    def solve(self, interval, dt):

        # Initial set-up
        (T0, T) = interval
        t0 = T0; t1 = T0 + dt
        vs0 = self.vs_

        vs_series = None
        u_series = None
        if self._parameters["store_solutions"]:
            vs_series = TimeSeries("results/vs_solutions")
            u_series = TimeSeries("results/u_solutions")

        while (t1 <= T + DOLFIN_EPS):
            # Solve
            info_blue("Solving on t = (%g, %g)" % (t0, t1))
            timestep = (t0, t1)
            self.step(timestep, vs0)

            if self._parameters["store_solutions"]:
                vs_series.store(self._domain, t1)
                theta = self._parameters["theta"]
                midt = t0 + theta*(t1 - t0)
                u_series.store(self._domain, midt)
                vs_series.store(self.vs.vector(), t1)
                u_series.store(self.u.vector(), midt)

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
        ode_timer1 = Timer("Tentative ODE step")
        begin("Tentative ODE step")
        vs_star = self.ode_step((t0, t), ics)
        end()
        ode_timer1.stop()
        (v_star, s_star) = split(vs_star)

        # Compute tentative potentials vu = (v, u)
        pde_timer = Timer("Tentative PDE step")
        begin("Tentative PDE step")
        vur = self.pde_step((t0, t1), vs_star)
        end()
        pde_timer.stop()
        (v, u, r) = split(vur)

        # Merge (inverse of split) v and s_star:
        v_s_star = utils.join((v, s_star), self.VS, annotate=annotate)

        # If first order splitting, we are essentially done:
        if theta == 1.0:
            self.vs.assign(v_s_star, annotate=annotate)
        # Otherwise, we do another ode_step:
        else:
            ode_timer2 = Timer("Corrective ODE step")
            begin("Corrective ODE step")
            vs = self.ode_step((t, t1), v_s_star)
            end()
            ode_timer2.stop()
            self.vs.assign(vs, annotate=annotate)

        # Update previous
        self.vs_.assign(self.vs)

        # Store u (Not a part of the solution algorithm, no need to
        # annotate, fortunately ..)
        self.u.assign(vur.split()[1], annotate=False)

        if self._parameters["plot_solutions"]:
            (v, s) = self.vs.split(deepcopy=True)
            self._v.assign(v, annotate=False)
            self._s.assign(s, annotate=False)
            plot(self._v, title="v")
            plot(self._s, title="s")
            plot(self.u, title="u")

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

        # Add current if applicable
        cell_current = self._model.cell_model().applied_current
        if cell_current:
            t = t0 + theta*(t1 - t0)
            cell_current.t = t
            I_theta += cell_current

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

    def pde_step(self, interval, vs_):
        """
        Solve

        v_t - div(M_i grad(v) ..) = 0
        div (M_i grad(v) + ..) = 0

        with v(t0) = v_,
        """

        # Hack, not sure if this is a good design
        (v_, s_) = split(vs_)

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
        solver.parameters["linear_solver"] = "cg"
        solver.parameters["preconditioner"] = "amg"
        solver.solve(annotate=annotate)
        return vur

class SplittingSolver(BasicSplittingSolver):
    """Optimized splitting solver for the bidomain equations"""

    def __init__(self, model, parameters=None):
        BasicSplittingSolver.__init__(self, model, parameters)

        # Define forms for pde_step
        self._k_n = Constant(-1.0)
        (self._a, self._L) = self.pde_variational_problem(self._k_n, self.vs_)

        # Pre-assemble left-hand side (will be updated if time-step
        # changes)
        self._A = assemble(self._a)

        # Tune solver types
        solver_type = self._parameters["linear_pde_solver"]
        if solver_type == "direct":
            self._linear_solver = LUSolver(self._A)
            self._linear_solver.parameters["same_nonzero_pattern"] = True
        elif solver_type == "iterative":
            self._linear_solver = KrylovSolver("cg", "amg")
            self._linear_solver.set_operator(self._A)
            self._linear_solver.parameters["preconditioner"]["same_nonzero_pattern"] = True

        else:
            error("Unknown linear_pde_solver specified: %s" % solver_type)

    def pde_variational_problem(self, k_n, vs_):

        # Extract conductivities from model
        M_i, M_e = self._model.conductivities()

        # Define variational formulation
        (v, u, r) = TrialFunctions(self.VUR)
        (w, q, s) = TestFunctions(self.VUR)

        # Extract theta parameter
        theta = self._parameters["theta"]

        # Set-up variational problem
        (v_, s_) = split(vs_)
        Dt_v = (v - v_)
        theta_parabolic = (theta*inner(M_i*grad(v), grad(w))*dx
                           + (1.0 - theta)*inner(M_i*grad(v_), grad(w))*dx
                           + inner(M_i*grad(u), grad(w))*dx)
        theta_elliptic = (theta*inner(M_i*grad(v), grad(q))*dx
                          + (1.0 - theta)*inner(M_i*grad(v_), grad(q))*dx
                          + inner((M_i + M_e)*grad(u), grad(q))*dx)

        G = (Dt_v*w*dx + k_n*theta_parabolic + theta_elliptic
             + (s*u + r*q)*dx)

        # Add applied current if specified
        if self._model.applied_current:
            G -= k_n*self._model.applied_current*w*dx

        (a, L) = system(G)
        return (a, L)

    def pde_step(self, interval, vs_):
        """
        Solve

        v_t - div(M_i grad(v) ..) = applied_current
        div (M_i grad(v) + ..) = 0

        with v(t0) = v_,
        """

        # Extract interval and time-step
        (t0, t1) = interval
        dt = (t1 - t0)
        theta = self._parameters["theta"]
        t = t0 + theta*dt

        annotate = self._parameters["enable_adjoint"]

        # Update previous solution
        self.vs_.assign(vs_)

        # Reuse as much as possible if possible
        solver_type = self._parameters["linear_pde_solver"]
        if dt == float(self._k_n):
            A = self._A
            if solver_type == "direct":
                info("Reusing LU factorization")
                self._linear_solver.parameters["reuse_factorization"] = True
            elif solver_type == "iterative":
                info("Reusing KrylovSolver preconditioner")
                self._linear_solver.parameters["preconditioner"]["reuse"] = True
            else:
                pass
        else:
            self._k_n.assign(Constant(dt))
            A = assemble(self._a)
            self._A = A
            self._linear_solver.set_operator(self._A)
            if solver_type == "direct":
                self._linear_solver.parameters["reuse_factorization"] = False
            elif solver_type == "iterative":
                self._linear_solver.parameters["preconditioner"]["reuse"] =False
            else:
                pass

        if self._model.applied_current:
            self._model.applied_current.t = t

        b = assemble(self._L)

        # Solve system
        vur = Function(self.VUR)

        self._linear_solver.solve(vur.vector(), b)
        return vur
