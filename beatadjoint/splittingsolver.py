"This module contains splitting solvers for (subclasses of) CardiacModel."

# Copyright (C) 2012 Marie E. Rognes (meg@simula.no)
# Use and modify at will
# Last changed: 2012-10-27

__all__ = ["SplittingSolver", "BasicSplittingSolver"]

from dolfin import *
from dolfin_adjoint import *
from beatadjoint import CardiacModel
from beatadjoint.utils import join

class BasicSplittingSolver:
    """Operator splitting based solver for a cardiac model.

    The splitting algorithm can be controlled by the parameter
    'theta'.  theta = 1.0 corresponds to a (1st order) Godunov
    splitting, theta = 0.5 to a (2nd order) Strang splitting.

    See p. 78 ff in Sundnes et al 2006 for details.

    Assumes that conductivities does not change over time.
    """
    def __init__(self, model, parameters=None):
        "Create solver from given Cardiac Model and (optional) parameters."

        assert isinstance(model, CardiacModel), \
            "Expecting CardiacModel as first argument"

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
        l = self.parameters["ode_polynomial_degree"]
        fam = self.parameters["ode_polynomial_family"]
        num_states = self._model.cell_model().num_states()

        self.V = FunctionSpace(domain, "CG", k)
        R = FunctionSpace(domain, "R", 0)
        self.VUR = MixedFunctionSpace([self.V, self.V, R])
        if num_states > 1:
            self.S = VectorFunctionSpace(domain, fam, l, num_states)
        else:
            self.S = FunctionSpace(domain, fam, l)
        self.VS = self.V*self.S

        # Helper functions
        self.u = Function(self.VUR.sub(1).collapse(), name="u")
        self.vs_ = Function(self.VS, name="vs_")
        self.vs = Function(self.VS, name="vs")

    @staticmethod
    def default_parameters():
        "Set-up and return default parameters."
        parameters = Parameters("SplittingSolver")
        parameters.add("enable_adjoint", False)
        parameters.add("theta", 0.5)
        parameters.add("default_timestep", 1.0)

        parameters.add("potential_polynomial_degree", 1)
        parameters.add("ode_polynomial_degree", 0)
        parameters.add("ode_polynomial_family", "DG")
        parameters.add("ode_theta", 0.5)
        parameters.add("num_threads", 0)

        parameters.add("use_avg_u_constraint", True)

        ode_solver_params = NonlinearVariationalSolver.default_parameters()
        parameters.add(ode_solver_params)

        pde_solver_params = LinearVariationalSolver.default_parameters()
        parameters.add(pde_solver_params)
        return parameters

    def solution_fields(self):
        "Return tuple of: (previous vs, current vs, and u)"
        return (self.vs_, self.vs, self.u)

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
        adj_start_timestep(t0)
        while (t1 <= T + DOLFIN_EPS):

            info_blue("Solving on t = (%g, %g)" % (t0, t1))
            timestep = (t0, t1)
            (vs, u) = self.step(timestep, self.vs_)
            self.vs.assign(vs, annotate=annotate)
            self.u.assign(u, annotate=False) # Not part of solution algorithm

            # Yield current solutions
            yield (timestep, self.vs, self.u)

            # Update previous and timetime
            finished = (t0 + dt) >= T
            self.vs_.assign(self.vs, annotate=annotate)
            adj_inc_timestep(time=t1, finished=finished)
            t0 = t1
            t1 = t0 + dt

    def step(self, interval, ics):
        "Step through given 'interval' with given initial conditions."

        # Extract some parameters for readability
        theta = self.parameters["theta"]
        annotate = self.parameters["enable_adjoint"]

        # Extract time domain
        (t0, t1) = interval
        dt = (t1 - t0)
        t = t0 + theta*dt

        # Compute tentative membrane potential and state (vs_star)
        begin("Tentative ODE step")
        vs_star = self.ode_step((t0, t), ics)
        end()

        # Compute tentative potentials vu = (v, u)
        begin("PDE step")
        vur = self.pde_step((t0, t1), vs_star)
        end()

        # Merge (inverse of split) v and s_star: (needed for adjointability)
        begin("Merging step")
        v = split(vur)[0]
        #(v, u, r) = split(vur)
        (v_star, s_star) = split(vs_star)
        v_s_star = join((v, s_star), self.VS, annotate=annotate,
                        solver_type="cg")
        end()

        # If first order splitting, we are done:
        if theta == 1.0:
            return (v_s_star, vur.split()[1])

        # Otherwise, we do another ode_step:
        begin("Corrective ODE step")
        vs = self.ode_step((t, t1), v_s_star)
        end()

        return (vs, vur.split()[1])

    def ode_step(self, interval, ics):
        "..."
        # For now, just use theta scheme. To be improved.

        # Extract time domain
        (t0, t1) = interval
        k_n = Constant(t1 - t0)

        # Extract initial conditions
        (v_, s_) = split(ics)

        # Set-up current variables
        vs = Function(self.VS, name="ode_vs")
        annotate = self.parameters["enable_adjoint"]
        vs.assign(ics, annotate=annotate) # Start with good guess
        (v, s) = split(vs)
        (w, r) = TestFunctions(self.VS)

        # Define equation based on cell model
        # Note sign for I_theta
        Dt_v = (v - v_)/k_n
        Dt_s = (s - s_)/k_n

        theta = self.parameters["ode_theta"]
        F = self._model.cell_model().F
        I_ion = self._model.cell_model().I
        I_theta = - (theta*I_ion(v, s) + (1 - theta)*I_ion(v_, s_))
        F_theta = theta*F(v, s) + (1 - theta)*F(v_, s_)

        # Add stimulus if applicable
        stimulus = self._model.stimulus
        if stimulus:
            t = t0 + theta*(t1 - t0)
            stimulus.t = t
            I_theta += stimulus

        # Set-up system
        G = (Dt_v - I_theta)*w*dx + inner(Dt_s - F_theta, r)*dx
        pde = NonlinearVariationalProblem(G, vs, J=derivative(G, vs))

        # Set-up solver
        parameters.num_threads = self.parameters["num_threads"]
        solver = NonlinearVariationalSolver(pde)
        solver_params = self.parameters["nonlinear_variational_solver"]
        solver.parameters.update(solver_params)

        # Solve system
        solver.solve(annotate=self.parameters["enable_adjoint"])
        parameters.num_threads = 0
        return vs

    def pde_step(self, interval, vs_):
        "..."

        # Hack, not sure if this is a good design (previous value for
        # s should not be required as data)
        (v_, s_) = split(vs_)

        # Extract interval and time-step
        (t0, t1) = interval
        k_n = Constant(t1 - t0)
        theta = self.parameters["theta"]
        annotate = self.parameters["enable_adjoint"]

        # Extract conductivities from model
        M_i, M_e = self._model.conductivities()

        # Define variational formulation
        (v, u, l) = TrialFunctions(self.VUR)
        (w, q, lamda) = TestFunctions(self.VUR)

        Dt_v = (v - v_)/k_n
        theta_parabolic = (theta*inner(M_i*grad(v), grad(w))*dx
                           + (1.0 - theta)*inner(M_i*grad(v_), grad(w))*dx
                           + inner(M_i*grad(u), grad(w))*dx)
        theta_elliptic = (theta*inner(M_i*grad(v), grad(q))*dx
                          + (1.0 - theta)*inner(M_i*grad(v_), grad(q))*dx
                          + inner((M_i + M_e)*grad(u), grad(q))*dx)
        G = (Dt_v*w*dx + theta_parabolic + theta_elliptic + (lamda*u + l*q)*dx)

        # Add applied current as source in ellipic equation is
        # applicable
        if self._model.applied_current:
            t = t0 + theta*(t1 - t0)
            self._model.applied_current.t = t
            G -= self._model.applied_current*q*dx

        # Define variational problem
        a, L = system(G)
        vur = Function(self.VUR, name="pde_vur")
        pde = LinearVariationalProblem(a, L, vur)

        # Set-up solver
        solver = LinearVariationalSolver(pde)
        solver_params = self.parameters["linear_variational_solver"]
        solver.parameters.update(solver_params)

        # Solve system
        solver.solve(annotate=annotate)

        return vur

class SplittingSolver(BasicSplittingSolver):
    """Optimized splitting solver for the bidomain equations"""

    def __init__(self, model, parameters=None):
        BasicSplittingSolver.__init__(self, model, parameters)

        # Define forms for pde_step
        self._k_n = Constant(self.parameters["default_timestep"])
        (self._a, self._L) = self.pde_variational_problem(self._k_n, self.vs_)

        # Pre-assemble left-hand side (will be updated if time-step
        # changes)
        self._A = assemble(self._a, annotate=self.parameters["enable_adjoint"])

        # Tune solver types
        solver_parameters = self.parameters["linear_variational_solver"]
        solver_type = solver_parameters["linear_solver"]
        if solver_type == "direct":
            self._linear_solver = LUSolver(self._A)
            self._linear_solver.parameters.update(solver_parameters["lu_solver"])
        elif solver_type == "iterative":
            self._linear_solver = KrylovSolver("gmres", "amg")
            self._linear_solver.parameters.update(solver_parameters["krylov_solver"])
            self._linear_solver.set_operator(self._A)
        else:
            error("Unknown linear_pde_solver specified: %s" % solver_type)

    @staticmethod
    def default_parameters():
        "Set-up and return default parameters."
        parameters = BasicSplittingSolver.default_parameters()

        # Customize linear solver parameters
        ps = parameters["linear_variational_solver"]
        ps["linear_solver"] = "iterative"
        ps["krylov_solver"]["preconditioner"]["same_nonzero_pattern"] = True
        ps["lu_solver"]["same_nonzero_pattern"] = True

        ps = parameters["nonlinear_variational_solver"]
        ps["linear_solver"] = "iterative"
        return parameters

    def linear_solver(self):
        "Return linear solver object (reused)."
        return self._linear_solver

    def pde_variational_problem(self, k_n, vs_):
        "Return left- and right-hand sides for variational problem"

        # Extract conductivities from model
        M_i, M_e = self._model.conductivities()

        # Extract theta parameter
        theta = self.parameters["theta"]

        # Define variational formulation
        use_R = self.parameters["use_avg_u_constraint"]
        if not use_R:
            self.VUR = MixedFunctionSpace([self.V, self.V])
            (v, u) = TrialFunctions(self.VUR)
            (w, q) = TestFunctions(self.VUR)
        else:
            (v, u, l) = TrialFunctions(self.VUR)
            (w, q, lamda) = TestFunctions(self.VUR)

        # Set-up variational problem
        (v_, s_) = split(vs_)
        Dt_v = (v - v_)

        theta_parabolic = (theta*inner(M_i*grad(v), grad(w))*dx
                           + (1.0 - theta)*inner(M_i*grad(v_), grad(w))*dx
                           + inner(M_i*grad(u), grad(w))*dx)
        theta_elliptic = (theta*inner(M_i*grad(v), grad(q))*dx
                          + (1.0 - theta)*inner(M_i*grad(v_), grad(q))*dx
                          + inner((M_i + M_e)*grad(u), grad(q))*dx)

        G = (Dt_v*w*dx + k_n*theta_parabolic + theta_elliptic)

        if use_R:
            G += (lamda*u + l*q)*dx

        # Add applied current if specified
        if self._model.applied_current:
            G -= k_n*self._model.applied_current*w*dx

        (a, L) = system(G)
        return (a, L)

    def pde_step(self, interval, vs_):
        "."

        # Extract interval and time-step
        (t0, t1) = interval
        dt = (t1 - t0)
        theta = self.parameters["theta"]
        t = t0 + theta*dt
        annotate = self.parameters["enable_adjoint"]

        # Update previous solution
        self.vs_.assign(vs_, annotate=annotate)

        # Assemble left-hand-side: only re-assemble if necessary, and
        # reuse all solver data possible
        solver = self.linear_solver()
        tolerance = 1.e-12
        if abs(dt - float(self._k_n)) < tolerance:
            A = self._A
            if isinstance(solver, LUSolver):
                info("Reusing LU factorization")
                solver.parameters["reuse_factorization"] = True
            elif isinstance(solver, KrylovSolver):
                info("Reusing KrylovSolver preconditioner")
                solver.parameters["preconditioner"]["reuse"] = True
            else:
                pass
        else:
            self._k_n.assign(Constant(dt))#, annotate=annotate) # FIXME
            A = assemble(self._a, annotate=annotate)
            self._A = A
            solver.set_operator(self._A)
            if isinstance(solver, LUSolver):
                solver.parameters["reuse_factorization"] = False
            elif isinstance(solver, KrylovSolver):
                solver.parameters["preconditioner"]["reuse"] = False
            else:
                pass

        # Assemble right-hand-side
        if self._model.applied_current:
            self._model.applied_current.t = t
        b = assemble(self._L, annotate=annotate)

        # Solve system
        vur = Function(self.VUR, name="pde_vur")
        solver.solve(vur.vector(), b, annotate=annotate)

        # Rescale u if KrylovSolver is used...
        if (isinstance(solver, KrylovSolver) and annotate==False):
            info_blue("Normalizing u")
            avg_u = assemble(split(vur)[1]*dx)
            bar = project(Constant((0.0, avg_u, 0.0)), self.VUR)
            vur.vector().axpy(-1.0, bar.vector())

        return vur

