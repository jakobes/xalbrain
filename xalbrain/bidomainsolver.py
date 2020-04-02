r"""These solvers solve the (pure) bidomain equations.

The equations are on the form: find the transmembrane potential :math:`v = v(x, t)` and
the extracellular potential :math:`u = u(x, t)` such that

.. math::

   v_t - \mathrm{div} ( G_i v + G_i u) = I_s

   \mathrm{div} (G_i v + (G_i + G_e) u) = I_a

where the subscript :math:`t` denotes the time derivative; :math:`G_x` denotes a weighted gradient: :math:`G_x = M_x \mathrm{grad}(v)` for
:math:`x \in \{i, e\}`, where :math:`M_i` and :math:`M_e` are the
intracellular and extracellular cardiac conductivity tensors,
respectively; :math:`I_s` and :math:`I_a` are prescribed input. In
addition, initial conditions are given for :math:`v`:

.. math::

   v(x, 0) = v_0

Finally, boundary conditions must be prescribed. For now, this solver
assumes pure homogeneous Neumann boundary conditions for :math:`v` and
:math:`u` and enforces the additional average value zero constraint
for u.
"""

# Copyright (C) 2013 Marie E. Rognes (meg@simula.no)
# Use and modify at will
# Last changed: 2013-04-18

from xalbrain.utils import time_stepper

import numpy as np
import dolfin as df
import typing as tp

from operator import or_
from functools import reduce

from abc import ABC


class AbstractBidomainSolver(ABC):
    r"""
    This solver is based on a theta-scheme discretization in time and FEM in space.

    .. note::

       For the sake of simplicity and consistency with other solver
       objects, this solver operates on its solution fields (as state
       variables) directly internally. More precisely, solve (and
       step) calls will act by updating the internal solution
       fields. It implies that initial conditions can be set (and are
       intended to be set) by modifying the solution fields prior to
       simulation.

    *Arguments*
      mesh (:py:class:`dolfin.Mesh`)
        The spatial domain (mesh)

      time (:py:class:`dolfin.Constant` or None)
        A constant holding the current time. If None is given, time is
        created for you, initialized to zero.

      M_i (:py:class:`ufl.Expr`)
        The intracellular conductivity tensor (as an UFL expression)

      M_e (:py:class:`ufl.Expr`)
        The extracellular conductivity tensor (as an UFL expression)

      I_s (:py:class:`dict`, optional)
        A typically time-dependent external stimulus given as a dict,
        with domain markers as the key and a
        :py:class:`dolfin.Expression` as values. NB: it is assumed
        that the time dependence of I_s is encoded via the 'time'
        Constant.

      I_a (:py:class:`dolfin.Expression`, optional)
        A (typically time-dependent) external applied current

      v\_ (:py:class:`ufl.Expr`, optional)
        Initial condition for v. A new :py:class:`dolfin.Function`
        will be created if none is given.

      parameters (:py:class:`dolfin.Parameters`, optional)
        Solver parameters
    """

    def __init__(
            self,
            mesh: df.Mesh,
            time: df.Constant,
            M_i: tp.Union[df.Expression, tp.Dict[int, df.Expression]],
            M_e: tp.Union[df.Expression, tp.Dict[int, df.Expression]],
            I_s: tp.Union[df.Expression, tp.Dict[int, df.Expression]] = None,
            I_a: tp.Union[df.Expression, tp.Dict[int, df.Expression]] = None,
            ect_current: tp.Dict[int, df.Expression] = None,
            v_: df.Function = None,
            cell_domains: df.MeshFunction = None,
            facet_domains: df.MeshFunction = None,
            dirichlet_bc: tp.List[tp.Tuple[df.Expression, int]] = None,
            dirichlet_bc_v: tp.List[tp.Tuple[df.Expression, int]] = None,
            parameters: df.Parameters = None
    ) -> None:
        """Initialise solverand check all parametersare correct."""
        self._timestep = None

        comm = df.MPI.comm_world
        rank = df.MPI.rank(comm)

        msg = "Expecting mesh to be a Mesh instance, not {}".format(mesh)
        assert isinstance(mesh, df.Mesh), msg

        msg = "Expecting time to be a Constant instance (or None)."
        assert isinstance(time, df.Constant) or time is None, msg

        msg = "Expecting parameters to be a Parameters instance (or None)"
        assert isinstance(parameters, df.Parameters) or parameters is None, msg

        self._nullspace_basis = None

        # Store input
        self._mesh = mesh
        self._time = time

        # Initialize and update parameters if given
        self._parameters = self.default_parameters()
        if parameters is not None:
            self._parameters.update(parameters)

        # Set-up function spaces
        k = self._parameters["polynomial_degree"]
        Ve = df.FiniteElement("CG", self._mesh.ufl_cell(), k)
        V = df.FunctionSpace(self._mesh, "CG", k)
        Ue = df.FiniteElement("CG", self._mesh.ufl_cell(), k)

        if self._parameters["linear_solver_type"] == "direct":
            Re = df.FiniteElement("R", self._mesh.ufl_cell(), 0)
            self.VUR = df.FunctionSpace(mesh, df.MixedElement((Ve, Ue, Re)))
        else:
            self.VUR = df.FunctionSpace(mesh, df.MixedElement((Ve, Ue)))

        self.V = V

        if cell_domains is None:
            cell_domains = df.MeshFunction("size_t", mesh, self._mesh.geometry().dim())
            cell_domains.set_all(0)

        # Chech that it is indeed a cell function.
        cell_dim = cell_domains.dim()
        mesh_dim = self._mesh.geometry().dim()
        msg = "Got {cell_dim}, expected {mesh_dim}.".format(cell_dim=cell_dim, mesh_dim=mesh_dim)
        assert cell_dim == mesh_dim, msg
        self._cell_domains = cell_domains

        if facet_domains is None:
            facet_domains = df.MeshFunction("size_t", mesh, self._mesh.geometry().dim() - 1)
            facet_domains.set_all(0)

        # Check that it is indeed a facet function.
        facet_dim = facet_domains.dim()
        msg = "Got {facet_dim}, expected {mesh_dim}.".format(
            facet_dim=facet_dim,
            mesh_dim=mesh_dim - 1
        )
        assert facet_dim == mesh_dim - 1, msg
        self._facet_domains = facet_domains

        # Set the intracellular conductivity
        cell_keys = set(self._cell_domains.array())
        all_cell_keys = comm.gather(cell_keys, root=0)
        if rank == 0:
            all_cell_keys = reduce(or_, all_cell_keys)
            if not isinstance(M_i, dict):
                M_i = {int(i): M_i for i in all_cell_keys}
            else:
                M_i_keys = set(M_i.keys())
                msg = "Got {M_i_keys}, expected {cell_keys}.".format(
                    M_i_keys=M_i_keys,
                    cell_keys=all_cell_keys
                )
                assert M_i_keys == all_cell_keys, msg

            if not isinstance(M_e, dict):
                M_e = {int(i): M_e for i in all_cell_keys}
            else:
                M_e_keys = set(M_e.keys())
                msg = "Got {M_e_keys}, expected {cell_keys}.".format(
                    M_e_keys=M_e_keys,
                    cell_keys=all_cell_keys
                )
                assert M_e_keys == all_cell_keys, msg
        else:
            M_i = None
            M_e = None

        self._M_i = comm.bcast(M_i, root=0)
        self._M_e = comm.bcast(M_e, root=0)
        assert M_i is not None, (M_i, rank)
        assert M_e is not None, (M_e, rank)

        # Store source terms
        self._I_s = I_s
        self._I_a = I_a

        # Set the ECT current, Note, it myst depend on `time` to be updated
        if ect_current is not None:
            ect_tags = set(ect_current.keys())
            facet_tags = set(self._facet_domains.array())
            msg = "{} not in facet domains ({}).".format(ect_tags, facet_tags)
            assert ect_tags <= facet_tags, msg
        self._ect_current = ect_current

        # Set-up solution fields:
        if v_ is None:
            self.merger = df.FunctionAssigner(V, self.VUR.sub(0))
            self.v_ = df.Function(V, name="v_")
        else:
            # df.debug("Experimental: v_ shipped from elsewhere.")
            self.merger = None
            self.v_ = v_
        self.vur = df.Function(self.VUR, name="vur")

        # Set Dirichlet bcs for the transmembrane potential
        self._bcs = []
        if dirichlet_bc_v is not None:
            for function, marker in dirichlet_bc_v:
                self._bcs.append(
                    df.DirichletBC(self.VUR.sub(0), function, self._facet_domains, marker)
                )

        # Set Dirichlet bcs for the extra cellular potential
        if dirichlet_bc is not None:
            for function, marker in dirichlet_bc:
                self._bcs.append(
                    df.DirichletBC(self.VUR.sub(1), function, self._facet_domains, marker)
                )

    @property
    def time(self) -> df.Constant:
        """The internal time of the solver."""
        return self._time

    def solution_fields(self) -> tp.Tuple[df.Function, df.Function]:
        """Return tuple of previous and current solution objects.

        Modifying these will modify the solution objects of the solver
        and thus provides a way for setting initial conditions for
        instance.

        Returns previous solution and current solution.
        """
        return self.v_, self.vur

    def solve(
        self,
        t0: float,
        t1: float,
        dt: float
    ) -> tp.Iterator[tp.Tuple[tp.Tuple[float, float], df.Function]]:
        """Solve the model on a time interval (`t0`, `t1`) with timestep `dt`.

        Arguments:
            t0: Start time.
            t1: End time.
            dt: Timestep.

        Returns the solution along with the valid time interval.

        *Example of usage*::

          # Create generator
          solutions = solver.solve((0.0, 1.0), 0.1)

          # Iterate over generator (computes solutions as you go)
          for (interval, solution_fields) in solutions:
            (t0, t1) = interval
            v_, vur = solution_fields
            # do something with the solutions
        """
        # Solve on entire interval if no interval is given.
        if dt is None:
            dt = t1 - t0

        # Step through time steps until at end time
        for _t0, _t1 in time_stepper(t0, t1, dt):
            self.step(_t0, _t1)

            # Yield solutions
            yield (_t0, _t1), self.solution_fields()

            # If not: update members and move to next time
            # Subfunction assignment would be good here.
            if isinstance(self.v_, df.Function):
                self.merger.assign(self.v_, self.vur.sub(0))

class BasicBidomainSolver(AbstractBidomainSolver):
    __doc__ = AbstractBidomainSolver.__doc__

    def step(self, t0: float, t1: float) -> None:
        """Solve on the given time interval (t0, t1).

        Arguments:
            interval (:py:class:`tuple`)
                The time interval (t0, t1) for the step

        *Invariants*
            Assuming that v\_ is in the correct state for t0, gives
            self.vur in correct state at t1.
        """
        timer = df.Timer("PDE step")

        # Extract theta and conductivities
        theta = self._parameters["theta"]
        Mi = self._M_i
        Me = self._M_e

        # Extract interval and thus time-step
        kn = df.Constant(t1 - t0)

        # Define variational formulation
        if self._parameters["linear_solver_type"] == "direct":
            v, u, l = df.TrialFunctions(self.VUR)
            w, q, lamda = df.TestFunctions(self.VUR)
        else:
            v, u = df.TrialFunctions(self.VUR)
            w, q = df.TestFunctions(self.VUR)

        # Get physical parameters
        chi = self._parameters["Chi"]
        capacitance = self._parameters["Cm"]

        Dt_v = (v - self.v_)/kn
        Dt_v *= chi*capacitance
        v_mid = theta*v + (1.0 - theta)*self.v_

        # Set time
        t = t0 + theta*(t1 - t0)
        self.time.assign(t)

        # Define spatial integration domains:
        dz = df.Measure("dx", domain=self._mesh, subdomain_data=self._cell_domains)
        db = df.Measure("ds", domain=self._mesh, subdomain_data=self._facet_domains)

        # Get domain labels
        cell_tags = map(int, set(self._cell_domains.array()))   # np.int64 does not workv
        facet_tags = map(int, set(self._facet_domains.array()))

        # Loop overe all domain labels
        G = Dt_v*w*dz()
        for key in cell_tags:
            G += df.inner(Mi[key]*df.grad(v_mid), df.grad(w))*dz(key)
            G += df.inner(Mi[key]*df.grad(u), df.grad(w))*dz(key)
            G += df.inner(Mi[key]*df.grad(v_mid), df.grad(q))*dz(key)
            G += df.inner((Mi[key] + Me[key])*df.grad(u), df.grad(q))*dz(key)

            if self._I_s is None:
                G -= chi*df.Constant(0)*w*dz(key)
            else:
                G -= chi*self._I_s*w*dz(key)

            # If Lagrangian multiplier
            if self._parameters["linear_solver_type"] == "direct":
                G += (lamda*u + l*q)*dz(key)

            # Add applied current as source in elliptic equation if applicable
            if self._I_a:
                G -= chi*self._I_a*q*dz(key)

        if self._ect_current is not None:
            for key in facet_tags:
                # Detfalt to 0 if not defined for that facet tag
                # TODO: Should I include `chi` here? I do not think so
                G += self._ect_current.get(key, df.Constant(0))*q*db(key)

        # Define variational problem
        a, L = df.system(G)
        pde = df.LinearVariationalProblem(a, L, self.vur, bcs=self._bcs)

        # Set-up solver
        solver = df.LinearVariationalSolver(pde)
        solver.solve()

    @staticmethod
    def default_parameters() -> df.Parameters:
        """Initialize and return a set of default parameters.

        To inspect all the default parameters, do::

          info(BasicBidomainSolver.default_parameters(), True)
        """
        parameters = df.Parameters("BasicBidomainSolver")
        parameters.add("theta", 0.5)
        parameters.add("polynomial_degree", 1)
        parameters.add("use_avg_u_constraint", True)

        # Set default solver type to be iterative
        parameters.add("linear_solver_type", "direct")

        # Set default iterative solver choices (used if iterative solver is invoked)
        parameters.add("algorithm", "gmres")
        parameters.add("preconditioner", "petsc_amg")

        parameters.add("Chi", 1.0)        # Membrane to volume ratio
        parameters.add("Cm", 1.0)         # Membrane capacitance
        return parameters


class BidomainSolver(AbstractBidomainSolver):
    __doc__ = AbstractBidomainSolver.__doc__

    @property
    def linear_solver(self) -> tp.Union[df.LinearVariationalSolver, df.PETScKrylovSolver]:
        """The linear solver (:py:class:`dolfin.LUSolver` or :py:class:`dolfin.PETScKrylovSolver`)."""
        return self._linear_solver

    def _create_linear_solver(self) -> None:
        """Helper function for creating linear solver based on parameters."""
        solver_type = self._parameters["linear_solver_type"]

        if solver_type == "direct":
            solver = df.LUSolver(self._lhs_matrix)

        elif solver_type == "iterative":
            # Initialize KrylovSolver with matrix
            alg = self._parameters["algorithm"]
            prec = self._parameters["preconditioner"]

            solver = df.PETScKrylovSolver(alg, prec)
            solver.set_operator(self._lhs_matrix)

            solver.parameters["nonzero_initial_guess"] = True

            # Set nullspace if present. We happen to know that the transpose nullspace is the same
            # as the nullspace (easy to prove from matrix structure).
            A = df.as_backend_type(self._lhs_matrix)
            A.set_nullspace(self.nullspace)
        else:
            df.error("Unknown linear_solver_type given: {}".format(solver_type))

        return solver

    @property
    def nullspace(self) -> df.VectorSpaceBasis:
        if self._nullspace_basis is None:
            null_vector = df.Vector(self.vur.vector())
            self.VUR.sub(1).dofmap().set(null_vector, 1.0)
            null_vector *= 1.0/null_vector.norm("l2")
            self._nullspace_basis = df.VectorSpaceBasis([null_vector])
        return self._nullspace_basis

    @staticmethod
    def default_parameters() -> df.Parameters:
        """Initialize and return a set of default parameters

        *Returns*
          A set of parameters (:py:class:`dolfin.Parameters`)

        To inspect all the default parameters, do::

          info(BidomainSolver.default_parameters(), True)
        """
        parameters = df.Parameters("BidomainSolver")
        parameters.add("theta", 0.5)
        parameters.add("polynomial_degree", 1)

        # Physical parameters
        parameters.add("Chi", 1.0)        # Membrane to volume ratio
        parameters.add("Cm", 1.0)         # Membrane capacitance

        # Set default solver type to be iterative
        parameters.add("linear_solver_type", "direct")
        parameters.add("use_avg_u_constraint", True)

        # Set default iterative solver choices (used if iterative solver is invoked)
        parameters.add("algorithm", "gmres")
        parameters.add("preconditioner", "petsc_amg")
        return parameters

    def variational_forms(self, kn: df.Constant) -> tp.Tuple[tp.Any, tp.Any]:
        """Create the variational forms corresponding to the given
        discretization of the given system of equations.

        *Arguments*
          kn (:py:class:`ufl.Expr` or float)
            The time step

        *Returns*
          (lhs, rhs) (:py:class:`tuple` of :py:class:`ufl.Form`)

        """
        # Extract theta parameter and conductivities
        theta = self._parameters["theta"]
        Mi = self._M_i
        Me = self._M_e

        # Define variational formulation
        if self._parameters["linear_solver_type"] == "direct":
            v, u, l = df.TrialFunctions(self.VUR)
            w, q, lamda = df.TestFunctions(self.VUR)
        else:
            v, u = df.TrialFunctions(self.VUR)
            w, q = df.TestFunctions(self.VUR)

        # Get physical parameters
        chi = self._parameters["Chi"]
        capacitance = self._parameters["Cm"]

        Dt_v = (v - self.v_)/kn
        Dt_v *= chi*capacitance
        v_mid = theta*v + (1.0 - theta)*self.v_

        # Set-up measure and rhs from stimulus
        dz = df.Measure("dx", domain=self._mesh, subdomain_data=self._cell_domains)
        db = df.Measure("ds", domain=self._mesh, subdomain_data=self._facet_domains)

        # Get domain tags
        cell_tags = map(int, set(self._cell_domains.array()))   # np.int64 does not work
        facet_tags = map(int, set(self._facet_domains.array()))

        # Loop over all domains
        G = Dt_v*w*dz()
        for key in cell_tags:
            G += df.inner(Mi[key]*df.grad(v_mid), df.grad(w))*dz(key)
            G += df.inner(Mi[key]*df.grad(u), df.grad(w))*dz(key)
            G += df.inner(Mi[key]*df.grad(v_mid), df.grad(q))*dz(key)
            G += df.inner((Mi[key] + Me[key])*df.grad(u), df.grad(q))*dz(key)

            if  self._I_s is None:
                G -= chi*df.Constant(0)*w*dz(key)
            else:
                G -= chi*self._I_s*w*dz(key)

            # If Lagrangian multiplier
            if self._parameters["linear_solver_type"] == "direct":
                G += (lamda*u + l*q)*dz(key)

            if self._I_a:
                G -= chi*self._I_a*q*dz(key)

        for key in facet_tags:
            if self._ect_current is not None:
                # Default to 0 if not defined for tag I do not I should apply `chi` here.
                G += self._ect_current.get(key, df.Constant(0))*q*db(key)

        a, L = df.system(G)
        return a, L

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
        theta = self._parameters["theta"]
        t = t0 + theta*dt
        self.time.assign(t)

        # Update matrix and linear solvers etc as needed
        if self._timestep is None:
            self._timestep = df.Constant(dt)
            self._lhs, self._rhs = self.variational_forms(self._timestep)

            # Preassemble left-hand side and initialize right-hand side vector
            self._lhs_matrix = df.assemble(self._lhs, keep_diagonal=True)     # TODO: Why diagonal?
            self._rhs_vector = df.Vector(self._mesh.mpi_comm(), self._lhs_matrix.size(0))
            self._lhs_matrix.init_vector(self._rhs_vector, 0)

            # Create linear solver (based on parameter choices)
            self._linear_solver = self._create_linear_solver()
        else:
            self._update_solver(dt)

        # Assemble right-hand-side
        df.assemble(self._rhs, tensor=self._rhs_vector)

        # Apply BCs
        for bc in self._bcs:
            bc.apply(self._lhs_matrix, self._rhs_vector)



        extracellular_indices = np.arange(0, self._rhs_vector.local_size(), 2)
        rhs_norm = self._rhs_vector.get_local()[extracellular_indices].sum()
        rhs_norm /= extracellular_indices.size
        # rhs_norm = self._rhs_vector.array()[extracellular_indices].sum()/extracellular_indices.size
        self._rhs_vector.get_local()[extracellular_indices] -= rhs_norm

        # Solve problem
        self.linear_solver.solve(
            self.vur.vector(),
            self._rhs_vector
        )

    def _update_solver(self, dt: tp.Union[float, df.Constant]) -> None:
        """Helper function for updating a KrylovSolver depending on whether timestep has changed."""
        if abs(dt - float(self._timestep)) < 1.e-12:
            return

        # Update time step
        self._timestep.assign(dt)

        # Reassemble matrix
        df.assemble(self._lhs, tensor=self._lhs_matrix)

        # Make new Krylov solver
        self._linear_solver = self._create_linear_solver()
