from dolfin import *

class SplittingSolver:
    """Operator splitting based solver for the bidomain equations."""
    def __init__(self, model, parameters=None):
        "Create solver."

        self._model = model

        # Set parameters
        self._parameters = self.default_parameters()
        if parameters is not None:
            self._parameters.add(parameters)

        # Extract theta parameter
        self._theta = self._parameters["theta"]

        # Create function spaces
        domain = self._model.mesh()
        k = self._parameters["potential_polynomial_degree"]
        self.W = VectorFunctionSpace(domain, "CG", k, 2)

        q = self._parameters["ode_polynomial_degree"]
        n = self._model.cell_model().num_states()
        self.S = VectorFunctionSpace(domain, "DG", q, n)

        # Create solution fields
        self.uv = Function(self.W)
        self.v = Function(self.W.sub(0).collapse())
        self.u = Function(self.W.sub(1).collapse())
        self.s = Function(self.S)

    def default_parameters(self):

        parameters = Parameters("SplittingSolver")
        parameters.add("theta", 0.5)
        parameters.add("potential_polynomial_degree", 1)
        parameters.add("ode_polynomial_degree", 0)

        return parameters

    def solution_fields(self):
        return (self.v, self.u, self.s)

    def step(self, interval, ics):
        "Step through given interval with given initial conditions"

        theta = self._theta

        # Extract time domain
        (t0, t1) = interval
        dt = (t1 - t0)
        t = t0 + theta*dt

        # Extract initial conditions
        (v_, s_) = ics

        # Compute tentative membrane potential (v_star) and state (s_star)
        (v_star, s_star) = self.ode_step((t0, t), (v_, s_))

        # Compute tentative potentials (v, u)
        (v, u) = self.pde_step((t0, t1), v_star)

        # Compute final membrane potential and state (if not done)
        if t < t1:
            s = s_star
        else:
            (v, s) = self.ode_step((t, t1), (v, s_star))

        return (v, s)

    def ode_step(self, interval, ics):
        """
        Solve

        v_t = - I_ion(v, s)
        s_t = F(v, s)

        with v(t0) = v_, s(t0) = s_
        """

        # Extract time domain
        (t0, t1) = interval

        # Extract initial conditions
        (v_, s_) = ics

        # Solve system here
        return (v_, s_)

    def pde_step(self, interval, ics):
        """
        Solve

        v_t - div(M_i grad(v) ...) = 0
        div (M_i grad(v) + ...) = 0

        with v(t0) = v_,
        """

        v_ = ics
        return (v_, None)


