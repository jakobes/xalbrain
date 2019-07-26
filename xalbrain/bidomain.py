import dolfin as df
import numpy as np

from typing import (
    Dict,
    Tuple,
    Any,
    Iterator,
    Union,
    Sequence,
)

from coupled_utils import (
    CellTags,
    InterfaceTags,
    BidomainParameters,
    time_stepper,
)


class BidomainSolver:
    def __init__(
        self,
        time: df.Constant,
        mesh: df.Mesh,
        intracellular_conductivity: Dict[int, df.Expression],
        extracellular_conductivity: Dict[int, df.Expression],
        cell_function: df.MeshFunction,
        cell_tags: CellTags,
        interface_function: df.MeshFunction,
        interface_tags: InterfaceTags,
        parameters: BidomainParameters,
        neumann_boundary_condition: Dict[int, df.Expression] = None,
        v_prev: df.Function = None,
        surface_to_volume_factor: Union[float, df.Constant] = None,
        membrane_capacitance: Union[float, df.Constant] = None,
    ) -> None:
        self._time = time
        self._mesh = mesh
        self._parameters = parameters

        # Strip none from cell tags
        cell_tags = set(cell_tags) - {None}

        if surface_to_volume_factor is None:
            surface_to_volume_factor = df.constant(1)

        if membrane_capacitance is None:
            membrane_capacitance = df.constant(1)

        # Set Chi*Cm
        self._chi_cm = df.Constant(surface_to_volume_factor)*df.Constant(membrane_capacitance)

        if not set(intracellular_conductivity.keys()) == {*tuple(extracellular_conductivity.keys())}:
            raise ValueError("intracellular conductivity and lambda does not havemnatching keys.")
        if not set(cell_tags) == set(intracellular_conductivity.keys()):
            raise ValueError("Cell tags does not match conductivity keys")
        self._intracellular_conductivity = intracellular_conductivity
        self._extracellular_conductivity = extracellular_conductivity

        # Check cell tags
        _cell_function_tags = set(cell_function.array())
        if set(cell_tags)!= _cell_function_tags:       # If not equal
            msg = "Mismatching cell tags. Expected {}, got {}"
            raise ValueError(msg.format(set(cell_tags), _cell_function_tags))
        self._cell_tags = set(cell_tags)
        self._cell_function = cell_function

        restrict_tags = self._parameters.restrict_tags
        if set(restrict_tags) >= self._cell_tags:
            msg = "restrict tags ({})is not a subset of cell tags ({})"
            raise ValueError(msg.format(set(restrict_tags), self._cell_tags))
        self._restrict_tags = set(restrict_tags)

        # Check interface tags
        _interface_function_tags = {*set(interface_function.array()), None}
        if not set(interface_tags) <= _interface_function_tags:     # if not subset of
            msg = "Mismatching interface tags. Expected {}, got {}"
            raise ValueError(msg.format(set(interface_tags), _interface_function_tags))
        self._interface_function = interface_function
        self._interface_tags = interface_tags

        # Set up function spaces
        self._transmembrane_function_space = df.FunctionSpace(self._mesh, "CG", 1)
        transmembrane_element = df.FiniteElement("CG", self._mesh.ufl_cell(), 1)
        extracellular_element = df.FiniteElement("CG", self._mesh.ufl_cell(), 1)

        if neumann_boundary_condition is None:
            self._neumann_bc: Dict[int, df.Expression] = dict()
        else:
            self._neumann_bc = neumann_boundary_condition

        if self._parameters.linear_solver_type == "direct":
            lagrange_element = df.FiniteElement("R", self._mesh.ufl_cell(), 0)
            mixed_element = df.MixedElement((transmembrane_element, extracellular_element, lagrange_element))
        else:
            mixed_element = df.MixedElement((transmembrane_element, extracellular_element))
        self._VUR = df.FunctionSpace(mesh, mixed_element)    # TODO: rename to something sensible

        # Set-up solution fields:
        if v_prev is None:
            self._merger = df.FunctionAssigner(self._transmembrane_function_space, self._VUR.sub(0))
            self._v_prev = df.Function(self._transmembrane_function_space, name="v_prev")
        else:
            self._merger = None
            self._v_prev = v_prev
        self._vur = df.Function(self._VUR, name="vur")        # TODO: Give sensible name

        # For normlising rhs_vector. TODO: Unsure about this. Check the nullspace from cbcbeat
        self._extracellular_dofs = np.asarray(self._VUR.sub(1).dofmap().dofs())

        # Mark first timestep
        self._timestep: df.Constant = None

    def solution_fields(self) -> Tuple[df.Function, df.Function]:
        """
        Return tuple of previous and current solution objects.

        Modifying these will modify the solution objects of the solver
        and thus provides a way for setting initial conditions for
        instance.

        *Returns*
          (previous v, current vur) (:py:class:`tuple` of :py:class:`dolfin.Function`)
        """
        return self._v_prev, self._vur

    def _create_linear_solver(self):
        """Helper function for creating linear solver based on parameters."""
        solver_type = self._parameters.linear_solver_type

        if solver_type == "direct":
            solver = df.LUSolver(self._lhs_matrix)
        elif solver_type == "iterative":
            alg = self._parameters.krylov_method
            prec = self._parameters.krylov_preconditioner

            solver = df.PETScKrylovSolver(alg, prec)
            solver.set_operator(self._lhs_matrix)
            solver.parameters["nonzero_initial_guess"] = True

            A = df.as_backend_type(self._lhs_matrix)
            A.set_nullspace(self._nullspace())
        else:
            msg = "Unknown solver type. Got {}, expected 'iterative' or 'direct'".format(solver_type)
            raise ValueError(msg)
        return solver

    def _nullspace(self) -> df.VectorSpaceBasis:
        null_vector = df.Vector(self._vur.vector())
        self._VUR.sub(1).dofmap().set(null_vector, 1.0)
        null_vector *= 1.0/null_vector.norm("l2")
        nullspace_basis = df.VectorSpaceBasis([null_vector])
        return nullspace_basis

    def variational_forms(self, dt: df.Constant) -> Tuple[Any, Any]:
        """Create the variational forms corresponding to the given
        discretization of the given system of equations.

        *Arguments*
          kn (:py:class:`ufl.Expr` or float)
            The time step

        *Returns*
          (lhs, rhs) (:py:class:`tuple` of :py:class:`ufl.Form`)

        """
        # Extract theta parameter and conductivities
        theta = self._parameters.theta
        Mi = self._intracellular_conductivity
        Me = self._extracellular_conductivity

        # Define variational formulation
        if self._parameters.linear_solver_type == "direct":
            v, u, multiplier = df.TrialFunctions(self._VUR)
            v_test, u_test, multiplier_test = df.TestFunctions(self._VUR)
        else:
            v, u = df.TrialFunctions(self._VUR)
            v_test, u_test = df.TestFunctions(self._VUR)

        Dt_v = (v - self._v_prev)/dt
        Dt_v *= self._chi_cm                # Chi is surface to volume aration. Cm is capacitance
        v_mid = theta*v + (1.0 - theta)*self._v_prev

        # Set-up measure and rhs from stimulus
        dOmega = df.Measure("dx", domain=self._mesh, subdomain_data=self._cell_function)
        dGamma = df.Measure("ds", domain=self._mesh, subdomain_data=self._interface_function)

        # Loop over all domains
        G = Dt_v*v_test*dOmega()
        for key in self._cell_tags - self._restrict_tags:
            G += df.inner(Mi[key]*df.grad(v_mid), df.grad(v_test))*dOmega(key)
            G += df.inner(Mi[key]*df.grad(v_mid), df.grad(u_test))*dOmega(key)

        for key in self._cell_tags:
            G += df.inner(Mi[key]*df.grad(u), df.grad(v_test))*dOmega(key)
            G += df.inner((Mi[key] + Me[key])*df.grad(u), df.grad(u_test))*dOmega(key)
            # If Lagrangian multiplier
            if self._parameters.linear_solver_type == "direct":
                G += (multiplier_test*u + multiplier*u_test)*dOmega(key)

        for key in set(self._interface_tags):
            # Default to 0 if not defined for tag
            G += self._neumann_bc.get(key, df.Constant(0))*u_test*dGamma(key)

        a, L = df.system(G)
        return a, L

    def solve(
            self,
            t0: float,
            t1: float,
            dt: float = None
    ) -> Iterator[Tuple[Tuple[float, float], Tuple[df.Function, df.Function]]]:
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
            self.step(*interval)
            yield interval, self.solution_fields()

            # TODO: Update wlsewhere?
            self._v_prev.assign(self._vur.sub(0))

    def step(self, t0: float, t1: float) -> None:
        r"""
        Solve on the given time step (t0, t1).

        *Arguments*
          interval (:py:class:`tuple`)
            The time interval (t0, t1) for the step

        *Invariants*
          Assuming that v\_ is in the correct state for t0, gives
          self.vur in correct state at t1.
        """
        dt = t1 - t0
        theta = self._parameters.theta
        t = t0 + theta*dt
        self._time.assign(t)

        # Update matrix and linear solvers etc as needed
        if self._timestep is None:
            self._timestep = df.Constant(dt)
            self._lhs, self._rhs = self.variational_forms(self._timestep)

            # Preassemble left-hand side and initialize right-hand side vector
            self._lhs_matrix = df.assemble(self._lhs, keep_diagonal=True)
            self._rhs_vector = df.Vector(self._mesh.mpi_comm(), self._lhs_matrix.size(0))
            # self._lhs_matrix.init_vector(self._rhs_vector, 0)
            self._lhs_matrix.ident_zeros()

            # Create linear solver (based on parameter choices)
            self._linear_solver = self._create_linear_solver()
        else:
            self._update_solver(dt)

        # Assemble right-hand-side
        df.assemble(self._rhs, tensor=self._rhs_vector)

        rhs_norm = self._rhs_vector[self._extracellular_dofs].sum()/self._extracellular_dofs.size
        self._rhs_vector[self._extracellular_dofs] -= rhs_norm

        # Solve problem
        self._linear_solver.solve(
            self._vur.vector(),
            self._rhs_vector
        )

    def _update_solver(self, dt: float) -> None:
        """Update the lhs matrix if timestep changes."""
        if (abs(dt - float(self._timestep)) < 1e-12):
            return
        self._timestep.assign(df.Constant(dt))

        # Reassemble matrix
        df.assemble(self._lhs, tensor=self._lhs_matrix)
