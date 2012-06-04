# Copyright (C) 2012 Marie E. Rognes (meg@simula.no)
# Use and modify at will
# Last changed: 2012-06-04

__all__ = ["CellSolver"]

import models
from dolfin import *
# try:
#     from dolfin_adjoint import *
# except:
#     print "dolfin_adjoint not found. Install it or mod this solver"
#     exit()

class CellSolver:
    """Simple ODE solver for just solving a cell model system (no
    space)."""

    def __init__(self, model, parameters=None):

        assert isinstance(model, models.CardiacCellModel), \
            "Expecting CardiacCellModel as first argument to CellSolver"

        # Set model and parameters
        self._model = model
        self._parameters = self.default_parameters()
        if parameters is not None:
            self._parameters.update(parameters)

        num_states = self._model.num_states()
        self._domain = UnitInterval(1)
        V = FunctionSpace(self._domain, "DG", 0)
        if num_states > 1:
            S = VectorFunctionSpace(self._domain, "DG", 0, num_states)
        else:
            S = FunctionSpace(self._domain, "DG", 0)
        self.VS = V*S

        # Helper functions
        self.vs_ = Function(self.VS)
        self.vs = Function(self.VS)

    def default_parameters(self):
        parameters = Parameters("CellSolver")
        parameters.add("theta", 0.5)
        return parameters

    def solution_fields(self):
        return (self.vs_, self.vs)

    def solve(self, interval, dt):

        # Initial set-up
        (T0, T) = interval
        t0 = T0; t1 = T0 + dt
        vs0 = self.vs_

        while (t1 <= T):
            # Solve
            info_blue("Solving on t = (%g, %g)" % (t0, t1))
            timestep = (t0, t1)
            vs = self.step(timestep, self.vs_)
            self.vs.assign(vs)

            # Update
            self.vs_.assign(self.vs)
            t0 = t1; t1 = t0 + dt

    def step(self, interval, ics):
        "Step through given interval with given initial conditions"

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
        Dt_v = (v - v_)/k_n
        Dt_s = (s - s_)/k_n

        theta = self._parameters["theta"]
        F = self._model.F
        I_ion = self._model.I
        I_theta = theta*I_ion(v, s) + (1 - theta)*I_ion(v_, s_)
        F_theta = theta*F(v, s) + (1 - theta)*F(v_, s_)

        # Set-up system
        G = (Dt_v - I_theta)*w*dx + inner(Dt_s - F_theta, r)*dx

        # Solve system
        pde = NonlinearVariationalProblem(G, vs, J=derivative(G, vs))
        solver = NonlinearVariationalSolver(pde)
        solver.solve()

        print "vs.vector() = ", vs.vector().array()
        return vs
