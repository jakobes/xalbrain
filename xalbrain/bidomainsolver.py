r"""
These solvers solve the (pure) bidomain equations.

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

# from xalbrain.dolfinimport import *
from xalbrain.markerwisefield import *
from xalbrain.utils import end_of_time, annotate_kwargs

import numpy as np

import dolfin as df

from typing import (
    Dict,
    Tuple,
    Union,
    Callable,
    List,
)


class BasicBidomainSolver:
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

      params (:py:class:`dolfin.Parameters`, optional)
        Solver parameters
    """

    def __init__(
            self,
            mesh: df.Mesh,
            time: df.Constant,
            M_i: Union[df.Expression, Dict[int, df.Expression]],
            M_e: Union[df.Expression, Dict[int, df.Expression]],
            I_s: Union[df.Expression, Dict[int, df.Expression]] = None,
            I_a: Union[df.Expression, Dict[int, df.Expression]] = None,
            ect_current: Dict[int, df.Expression]=None,
            v_: df.Function = None,
            cell_domains: df.MeshFunction = None,
            facet_domains: df.MeshFunction = None,
            dirichlet_bc: List[Tuple[df.Expression, df.MeshFunction, int]] = None,
            params: df.Parameters = None
    ) -> None:
        """Initialise solverand check all parametersare correct."""
        msg = "Expecting mesh to be a Mesh instance, not {}".format(mesh)
        assert isinstance(mesh, df.Mesh), msg

        msg = "Expecting time to be a Constant instance (or None)."
        assert isinstance(time, df.Constant) or time is None, msg

        msg = "Expecting params to be a Parameters instance (or None)"
        assert isinstance(params, df.Parameters) or params is None, msg

        self._nullspace_basis = None

        # Store input
        self._mesh = mesh
        self._time = time

        # Initialize and update parameters if given
        self.parameters = self.default_parameters()
        if params is not None:
            self.parameters.update(params)

        # Set-up function spaces
        k = self.parameters["polynomial_degree"]
        Ve = df.FiniteElement("CG", self._mesh.ufl_cell(), k)
        V = df.FunctionSpace(self._mesh, "CG", k)
        Ue = df.FiniteElement("CG", self._mesh.ufl_cell(), k)

        use_R = self.parameters["use_avg_u_constraint"]
        if use_R:
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
        assert cell_dim == mesh_dim, f"Got {cell_dim}, expected {mesh_dim}."
        self._cell_domains = cell_domains

        if facet_domains is None:
            facet_domains = df.MeshFunction("size_t", mesh, self._mesh.geometry().dim() - 1)
            facet_domains.set_all(0)

        # Check that it is indeed a facet function.
        facet_dim = facet_domains.dim()
        assert facet_dim == mesh_dim - 1, f"Got {facet_dim}, expected {mesh_dim - 1}."
        self._facet_domains = facet_domains

        if not isinstance(M_i, dict):
            M_i = {int(i): M_i for i in set(self._cell_domains.array())}
        else:
            M_i_keys = set(M_i.keys())
            cell_keys = set(self._cell_domains.array())
            assert M_i_keys == cell_keys, f"Got {M_i_keys}, expected {cell_keys}."
        self._M_i = M_i

        if not isinstance(M_e, dict):
            M_e = {int(i): M_e for i in set(self._cell_domains.array())}
        else:
            assert set(M_e.keys()) == set(self._cell_domains.array())
        self._M_e = M_e

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
            df.debug("Experimental: v_ shipped from elsewhere.")
            self.merger = None
            self.v_ = v_
        self.vur = df.Function(self.VUR, name="vur")

        # Set Dirichlet bcs
        self.bcs = []
        if dirichlet_bc is not None:
            for function, mesh_function, marker in dirichlet_bc:
                self.bcs.append(df.DirichletBC(self.VUR.sub(1), function,  mesh_function, marker))

    @property
    def time(self) -> df.Constant:
        """The internal time of the solver."""
        return self._time

    def solution_fields(self) -> Tuple[df.Function, df.Function]:
        """
        Return tuple of previous and current solution objects.

        Modifying these will modify the solution objects of the solver
        and thus provides a way for setting initial conditions for
        instance.

        *Returns*
          (previous v, current vur) (:py:class:`tuple` of :py:class:`dolfin.Function`)
        """
        return self.v_, self.vur


    def solve(self, interval: Tuple[float, float], dt: float=None) -> None:
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
          (timestep, solution_fields) via (:py:class:`genexpr`)

        *Example of usage*::

          # Create generator
          solutions = solver.solve((0.0, 1.0), 0.1)

          # Iterate over generator (computes solutions as you go)
          for (interval, solution_fields) in solutions:
            (t0, t1) = interval
            v_, vur = solution_fields
            # do something with the solutions
        """
        timer = df.Timer("PDE step")

        # Initial set-up
        # Solve on entire interval if no interval is given.
        T0, T = interval
        if dt is None:
            dt = T - T0
        t0 = T0
        t1 = T0 + dt

        # Step through time steps until at end time
        while True:
            df.info("Solving on t = ({:g}, {:g})".format(t0, t1))
            self.step((t0, t1))

            # Yield solutions
            yield (t0, t1), self.solution_fields()

            # Break if this is the last step
            if end_of_time(T, t0, t1, dt):
                break

            # If not: update members and move to next time
            # Subfunction assignment would be good here.

            if isinstance(self.v_, df.Function):
                self.merger.assign(self.v_, self.vur.sub(0))
            else:
                debug("Assuming that v_ is updated elsewhere. Experimental.")
            t0 = t1
            t1 = t0 + dt

    def step(self, interval: Tuple[float, float]) -> None:
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
        theta = self.parameters["theta"]
        Mi = self._M_i
        Me = self._M_e

        # Extract interval and thus time-step
        t0, t1 = interval
        k_n = df.Constant(t1 - t0)

        # Define variational formulation
        use_R = self.parameters["use_avg_u_constraint"]
        if use_R:
            v, u, l = df.TrialFunctions(self.VUR)
            w, q, lamda = df.TestFunctions(self.VUR)
        else:
            v, u = df.TrialFunctions(self.VUR)
            w, q = df.TestFunctions(self.VUR)

        Dt_v = (v - self.v_)/k_n
        v_mid = theta*v + (1.0 - theta)*self.v_

        # Set time
        t = t0 + theta*(t1 - t0)
        self.time.assign(t)

        # Define spatial integration domains:
        # (dz, rhs) = rhs_with_markerwise_field(self._I_s, self._mesh, w)

        dz = df.Measure("dx", domain=self._mesh, subdomain_data=self._cell_domains)
        db = df.Measure("ds", domain=self._mesh, subdomain_data=self._facet_domains)
        cell_tags = map(int, set(self._cell_domains.array()))   # np.int64 does not work
        facet_tags = map(int, set(self._facet_domains.array()))

        G = Dt_v*w*dz()
        for key in cell_tags:
            G += df.inner(Mi[key]*df.grad(v_mid), df.grad(w))*dz(key)
            G += df.inner(Mi[key]*df.grad(u), df.grad(w))*dz(key)
            G += df.inner(Mi[key]*df.grad(v_mid), df.grad(q))*dz(key)
            G += df.inner((Mi[key] + Me[key])*df.grad(u), df.grad(q))*dz(key)

            if self._I_s is None:
                G -= df.Constant(0)*w*dz(key)
            else:
                G -= self._I_s*w*dz(key)

            # If Lagrangian multiplier
            if use_R:
                G += (lamda*u + l*q)*dz(key)

            # Add applied current as source in elliptic equation if applicable
            if self._I_a:
                G -= self._I_a*q*dz(key)

        if self._ect_current is not None:
            for key in facet_tags:
                # Detfaltto 0 if not defined for that facet tag
                G += self._ect_current.get(key, df.Constant(0))*q*db(key)

        # Define variational problem
        a, L = df.system(G)

        pde = df.LinearVariationalProblem(a, L, self.vur)

        # Set-up solver
        solver = df.LinearVariationalSolver(pde)
        solver.parameters.update(self.parameters["linear_variational_solver"])
        solver.parameters["linear_solver"] = self.parameters["linear_solver_type"]
        solver.solve()

    @staticmethod
    def default_parameters() -> df.Parameters:
        """Initialize and return a set of default parameters

        *Returns*
          A set of parameters (:py:class:`dolfin.Parameters`)

        To inspect all the default parameters, do::

          info(BasicBidomainSolver.default_parameters(), True)
        """

        params = df.Parameters("BasicBidomainSolver")
        params.add("enable_adjoint", False)
        params.add("theta", 0.5)
        params.add("polynomial_degree", 1)
        params.add("use_avg_u_constraint", True)

        # Set default solver type to be iterative
        params.add("linear_solver_type", "direct")

        # Set default iterative solver choices (used if iterative
        # solver is invoked)
        params.add("algorithm", "gmres")
        params.add("preconditioner", "petsc_amg")

        # Add default parameters from both LU and Krylov solvers

        params.add(df.LUSolver.default_parameters())
        # Customize default parameters for LUSolver
        params["lu_solver"]["same_nonzero_pattern"] = True

        linear_params = df.LinearVariationalSolver.default_parameters()
        linear_params["krylov_solver"]["absolute_tolerance"] = 1e-14
        linear_params["krylov_solver"]["relative_tolerance"] = 1e-14
        linear_params["krylov_solver"]["nonzero_initial_guess"] = True
        params.add(linear_params)
        return params


class BidomainSolver(BasicBidomainSolver):
    __doc__ = BasicBidomainSolver.__doc__

    def __init__(
            self,
            mesh: df.Mesh,
            time: df.Constant,
            M_i: Union[df.Expression, Dict[int, df.Expression]],
            M_e: Union[df.Expression, Dict[int, df.Expression]],
            I_s: Union[df.Expression, Dict[int, df.Expression]] = None,
            I_a: Union[df.Expression, Dict[int, df.Expression]] = None,
            ect_current: Dict[int, df.Expression] = None,
            v_: df.Function = None,
            cell_domains: df.MeshFunction = None,
            facet_domains: df.MeshFunction = None,
            dirichlet_bc: List[Tuple[df.Expression, df.MeshFunction, int]] = None,
            params: df.Parameters = None
    ) -> None:
        # Call super-class
        BasicBidomainSolver.__init__(
            self,
            mesh,
            time,
            M_i,
            M_e,
            I_s=I_s,
            I_a=I_a,
            v_=v_,
            ect_current=ect_current,
            cell_domains=cell_domains,
            facet_domains=facet_domains,
            dirichlet_bc=dirichlet_bc,
            params=params
        )

        # Check consistency of parameters first
        if self.parameters["enable_adjoint"] and not dolfin_adjoint:
            df.warning("'enable_adjoint' is set to True, but no dolfin_adjoint installed.")

        # Mark the timestep as unset
        self._timestep = None

    @property
    def linear_solver(self) -> Union[df.LinearVariationalSolver, df.PETScKrylovSolver]:
        """The linear solver (:py:class:`dolfin.LUSolver` or
        :py:class:`dolfin.PETScKrylovSolver`)."""
        return self._linear_solver

    def _create_linear_solver(self):
        """Helper function for creating linear solver based on parameters."""
        solver_type = self.parameters["linear_solver_type"]

        if solver_type == "direct":
            solver = df.LUSolver(self._lhs_matrix)
            solver.parameters.update(self.parameters["lu_solver"])
            solver.parameters["reuse_factorization"] = True
            update_routine = self._update_lu_solver

        elif solver_type == "iterative":

            # Initialize KrylovSolver with matrix
            alg = self.parameters["algorithm"]
            prec = self.parameters["preconditioner"]

            df.debug("Creating PETSCKrylovSolver with %s and %s" % (alg, prec))

            solver = df.PETScKrylovSolver(alg, prec)
            solver.set_operator(self._lhs_matrix)

            # TODO: Espose these parameters
            solver.parameters.update(self.parameters["petsc_krylov_solver"])
            solver.parameters.convergence_norm_type = "preconditioned"
            solver.parameters.monitor_convergence = False
            solver.parameters.report = False
            solver.parameters.maximum_iterations = None
            solver.parameters.nonzero_initial_guess = True

            # Set nullspace if present. We happen to know that the
            # transpose nullspace is the same as the nullspace (easy
            # to prove from matrix structure).
            if self.parameters["use_avg_u_constraint"]:
                # Nothing to do, no null space
                pass
            else:
                # If dolfin-adjoint is enabled and installled: set the solver nullspace
                if dolfin_adjoint:
                    solver.set_nullspace(self.nullspace)
                    solver.set_transpose_nullspace(self.nullspace)
                # Otherwise, set the nullspace in the operator
                # directly.
                else:
                    A = as_backend_type(self._lhs_matrix)
                    A.set_nullspace(self.nullspace)

            update_routine = self._update_krylov_solver
        else:
            error("Unknown linear_solver_type given: %s" % solver_type)

        return solver, update_routine

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

        params = df.Parameters("BidomainSolver")
        params.add("enable_adjoint", False)
        params.add("theta", 0.5)
        params.add("polynomial_degree", 1)

        # Set default solver type to be iterative
        params.add("linear_solver_type", "direct")
        params.add("use_avg_u_constraint", True)

        # Set default iterative solver choices (used if iterative
        # solver is invoked)
        params.add("algorithm", "gmres")
        params.add("preconditioner", "petsc_amg")

        # Add default parameters from both LU and Krylov solvers
        params.add(df.LUSolver.default_parameters())
        petsc_params = df.PETScKrylovSolver.default_parameters()
        petsc_params["absolute_tolerance"] = 1e-14
        petsc_params["relative_tolerance"] = 1e-14
        petsc_params["nonzero_initial_guess"] = True
        params.add(petsc_params)

        # Customize default parameters for LUSolver
        params["lu_solver"]["same_nonzero_pattern"] = True
        return params

    def variational_forms(self, kn: df.Constant) -> Tuple[df.lhs, df.rhs]:
        """Create the variational forms corresponding to the given
        discretization of the given system of equations.

        *Arguments*
          k_n (:py:class:`ufl.Expr` or float)
            The time step

        *Returns*
          (lhs, rhs) (:py:class:`tuple` of :py:class:`ufl.Form`)

        """
        # Extract theta parameter and conductivities
        theta = self.parameters["theta"]
        Mi = self._M_i
        Me = self._M_e

        # Define variational formulation
        use_R = self.parameters["use_avg_u_constraint"]
        if use_R:
            v, u, l = df.TrialFunctions(self.VUR)
            w, q, lamda = df.TestFunctions(self.VUR)
        else:
            v, u = df.TrialFunctions(self.VUR)
            w, q = df.TestFunctions(self.VUR)


        Dt_v = (v - self.v_)/kn
        v_mid = theta*v + (1.0 - theta)*self.v_

        # Set-up measure and rhs from stimulus
        dz = df.Measure("dx", domain=self._mesh, subdomain_data=self._cell_domains)
        db = df.Measure("ds", domain=self._mesh, subdomain_data=self._facet_domains)
        cell_tags = map(int, set(self._cell_domains.array()))   # np.int64 does not work
        facet_tags = map(int, set(self._facet_domains.array()))

        G = Dt_v*w*dz()
        for key in cell_tags:
            G += df.inner(Mi[key]*df.grad(v_mid), df.grad(w))*dz(key)
            G += df.inner(Mi[key]*df.grad(u), df.grad(w))*dz(key)
            G += df.inner(Mi[key]*df.grad(v_mid), df.grad(q))*dz(key)
            G += df.inner((Mi[key] + Me[key])*df.grad(u), df.grad(q))*dz(key)

            if  self._I_s is None:
                G -= df.Constant(0)*w*dz(key)
            else:
                G -= self._I_s*w*dz(key)

            # If Lagrangian multiplier
            if use_R:
                G += (lamda*u + l*q)*dz(key)

            if self._I_a:
                G -= self._I_a*q*dz(key)

        for key in facet_tags:
            if self._ect_current is not None:
                # Default to 0 if not defined for tag
                G += self._ect_current.get(key, df.Constant(0))*q*db(key)

        a, L = df.system(G)
        return a, L

    def step(self, interval: Tuple[float, float]) -> None:
        """
        Solve on the given time step (t0, t1).

        *Arguments*
          interval (:py:class:`tuple`)
            The time interval (t0, t1) for the step

        *Invariants*
          Assuming that v\_ is in the correct state for t0, gives
          self.vur in correct state at t1.
        """

        timer = df.Timer("PDE step")
        solver_type = self.parameters["linear_solver_type"]

        # Extract interval and thus time-step
        t0, t1 = interval
        dt = t1 - t0
        theta = self.parameters["theta"]
        t = t0 + theta*dt
        self.time.assign(t)

        # Update matrix and linear solvers etc as needed
        if self._timestep is None:
            self._timestep = df.Constant(dt)
            self._lhs, self._rhs = self.variational_forms(self._timestep)

            # Preassemble left-hand side and initialize right-hand side vector
            df.debug("Preassembling bidomain matrix (and initializing vector)")
            self._lhs_matrix = df.assemble(self._lhs)
            self._rhs_vector = df.Vector(self._mesh.mpi_comm(), self._lhs_matrix.size(0))

            self._lhs_matrix.init_vector(self._rhs_vector, 0)

            # Create linear solver (based on parameter choices)
            self._linear_solver, self._update_solver = self._create_linear_solver()
        else:
            timestep_unchanged = abs(dt - float(self._timestep)) < 1.e-12
            self._update_solver(timestep_unchanged, dt)

        # Assemble right-hand-side
        df.assemble(self._rhs, tensor=self._rhs_vector)

        rhs_norm = self._rhs_vector.array()[:].sum()/self._rhs_vector.size()/2
        self._rhs_vector.array()[:] -= rhs_norm

        # Solve problem
        self.linear_solver.solve(
            self.vur.vector(),
            self._rhs_vector
        )

    def _update_lu_solver(self, timestep_unchanged: df.Constant, dt: df.Constant) -> None:
        """Helper function for updating an LUSolver depending on whether timestep has changed."""

        # Update reuse of factorization parameter in accordance with
        # changes in timestep
        if timestep_unchanged:
            df.debug("Timestep is unchanged, reusing LU factorization")
        else:
            df.debug("Timestep has changed, updating LU factorization")
            if dolfin_adjoint and self.parameters["enable_adjoint"]:
                raise ValueError("dolfin-adjoint doesn't support changing timestep (yet)")

            # Update stored timestep
            # FIXME: dolfin_adjoint still can't annotate constant assignment.
            self._timestep.assign(df.Constant(dt))#, annotate=annotate)

            # Reassemble matrix
            df.assemble(self._lhs, tensor=self._lhs_matrix)

            self._linear_solver, dummy = self._create_linear_solver()

    def _update_krylov_solver(self, timestep_unchanged: df.Constant, dt: df.Constant):
        """Helper function for updating a KrylovSolver depending on whether timestep has changed."""

        # Update reuse of preconditioner parameter in accordance with
        # changes in timestep
        if timestep_unchanged:
            df.debug("Timestep is unchanged, reusing preconditioner")
        else:
            df.debug("Timestep has changed, updating preconditioner")
            if dolfin_adjoint and self.parameters["enable_adjoint"]:
                raise ValueError("dolfin-adjoint doesn't support changing timestep (yet)")

            # Update stored timestep
            self._timestep.assign(df.Constant(dt))#, annotate=annotate)

            # Reassemble matrix
            df.assemble(self._lhs, tensor=self._lhs_matrix, **self._annotate_kwargs)

            # Make new Krylov solver
            self._linear_solver, dummy = self._create_linear_solver()

        # Set nonzero initial guess if it indeed is nonzero
        if self.vur.vector().norm("l2") > 1.e-12:
            debug("Initial guess is non-zero.")
            self.linear_solver.parameters["nonzero_initial_guess"] = True
