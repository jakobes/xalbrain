import dolfin as df

from typing import (
    Dict,
    Any,
    Tuple,
    NamedTuple,
    Iterable,
)

from xalbrain.utils import (
    time_stepper,
    create_linear_solver,
)


class MonodomainParameters(NamedTuple):
    timestep: df.Constant = df.Constant(1.0)
    theta: df.Constant = df.Constant(0.5)
    linear_solver_type: str = "direct"
    lu_type: str = "default"
    krylov_method: str = "cg"
    krylov_preconditioner: str = "petsc_amg"


class MonodomainSolver:
    def __init__(
        self,
        *,
        time: df.Constant,
        mesh: df.Mesh,
        conductivity: Dict[int, df.Expression],
        conductivity_ratio: Dict[int, df.Expression],
        parameters: MonodomainParameters,
        cell_function: df.MeshFunction = None,
        interface_function: df.MeshFunction = None,
        neumann_boundary_condition: Dict[int, df.Expression] = None,
        v_prev: df.Function = None
    ) -> None:
        self._time = time
        self._mesh = mesh
        self._parameters = parameters
        self._interface_function = interface_function

        if cell_function is None:
            cell_function = df.MeshFunction("size_t", mesh, mesh.geometric_dimension())
            cell_function.set_all(0)
        self._cell_function = cell_function
        self._cell_tags = set(self._cell_function.array())

        if interface_function is None:
            interface_function = df.MeshFunction("size_t", mesh, mesh.geometric_dimension() - 1)
            interface_function.set_all(0)
        self._interface_function = interface_function
        self._interface_tags = set(self._interface_function.array())

        if neumann_boundary_condition is None:
            self._neumann_boundary_condition: Dict[int, df.Expression] = dict()
        else:
            self._neumann_boundary_condition = neumann_boundary_condition

        try:
            if not set(conductivity.keys()) == set(conductivity_ratio.keys()):
                raise ValueError("intracellular conductivity and lambda does not have natching keys.")
        except AttributeError:
            conductivity = {k: conductivity for k in self._cell_tags}
            conductivity_ratio = {k: conductivity_ratio for k in self._cell_tags}
        self._conductivity = conductivity
        self._lambda = conductivity_ratio

        # Function spaces
        self._function_space = df.FunctionSpace(mesh, "CG", 1)

        # Test and trial and previous functions
        self._v_trial = df.TrialFunction(self._function_space)
        self._v_test = df.TestFunction(self._function_space)

        self._v = df.Function(self._function_space)
        if v_prev is None:
            self._v_prev = df.Function(self._function_space)
        else:
            # v_prev is shipped from an odesolver.
            self._v_prev = v_prev

        _cell_tags = set(self._cell_function.array())

        _interface_tags = set(self._interface_tags)
        _interface_function_values = {*set(self._interface_function.array()), None}
        if not _interface_tags <= _interface_function_values:
            msg = f"interface function does not contain {_interface_tags - _interface_function_values}."
            raise ValueError(msg)

        # Crete integration measures -- Interfaces
        self._dGamma = df.Measure("ds", domain=self._mesh, subdomain_data=self._interface_function)

        # Crete integration measures -- Cells
        self._dOmega = df.Measure("dx", domain=self._mesh, subdomain_data=self._cell_function)

        # Create variational forms
        self._timestep = df.Constant(self._parameters.timestep)
        self._lhs, self._rhs = self._variational_forms()

        # Preassemble left-hand side (will be updated if time-step changes)
        self._lhs_matrix = df.assemble(self._lhs)
        self._rhs_vector = df.Vector(mesh.mpi_comm(), self._lhs_matrix.size(0))
        self._lhs_matrix.init_vector(self._rhs_vector, 0)

        self._linear_solver = create_linear_solver(self._lhs_matrix, self._parameters)

    def _variational_forms(self) -> Tuple[Any, Any]:
        # Localise variables for convenicence
        dt = self._timestep
        theta = self._parameters.theta
        Mi = self._conductivity
        lbda = self._lambda

        dOmega = self._dOmega
        dGamma = self._dGamma

        v = self._v_trial
        v_test = self._v_test

        # Set-up variational problem
        dvdt = (v - self._v_prev)/dt
        v_mid = theta*v + (1.0 - theta)*self._v_prev

        # Cell contributions
        Form = dvdt*v_test*dOmega()
        for cell_tag in self._cell_tags:
            Form += df.inner(Mi[cell_tag]*df.grad(v_mid), df.grad(v_test))*dOmega(cell_tag)

        # Boundary contributions
        for interface_tag in self._interface_tags:
            neumann_bc = self._neumann_boundary_condition.get(interface_tag, df.Constant(0))
            neumann_bc = neumann_bc*v_test*dGamma(interface_tag)
            Form += neumann_bc

        a, L = df.system(Form)
        return a, L

    def solution_fields(self) -> Tuple[df.Function, df.Function]:
        """Return current and previous solution."""
        return self._v_prev, self._v

    def step(self, t0, t1) -> None:
        # Extract interval and thus time-step
        theta = self._parameters.theta
        dt = t1 - t0
        t = t0 + theta*dt
        self._time.assign(t)

        # Update matrix and linear solvers etc as needed
        self._update_solver(dt)

        # Assemble right-hand-side
        df.assemble(self._rhs, tensor=self._rhs_vector)

        # Solve problem
        self._linear_solver.solve(
            self._v.vector(),
            self._rhs_vector
        )

    def solve(
            self,
            t0: float,
            t1: float,
            dt: float = None
    ) -> Iterable[Tuple[Tuple[float, float], Tuple[df.Function, df.Function]]]:
        """
        Solve the discretization on a given time interval (t0, t1)
        with a given timestep dt and return generator for a tuple of
        the interval and the current solution.

        *Arguments*
          interval (:py:class:`tuple`)
            The time interval for the solve given by (t0, t1)
          dt (int, optional)
            The timestep for the solve. Defaults to length of interval

        *Returns*
          (timestep, solution_field) via (:py:class:`genexpr`)

        *Example of usage*::

          # Create generator
          solutions = solver.solve((0.0, 1.0), 0.1)

          # Iterate over generator (computes solutions as you go)
          for (interval, solution_fields) in solutions:
            (t0, t1) = interval
            v_, v = solution_fields
            # do something with the solutions
        """
        for interval in time_stepper(t0=t0, t1=t1, dt=dt):
            # info("Solving on t = (%g, %g)" % (t0, t1))
            self.step(interval)

            # Yield solutions
            yield interval, self.solution_fields()

            # Update wlsewhere???
            self._v_prev.assign(self._v)

    def _update_solver(self, dt: float) -> None:
        """Update the lhs matrix if timestep changes."""
        if (abs(dt - float(self._timestep)) < 1e-12):
            return
        self._timestep.assign(df.Constant(dt))

        # Reassemble matrix
        df.assemble(self._lhs, tensor=self._lhs_matrix)


