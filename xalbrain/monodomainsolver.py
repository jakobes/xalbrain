r"""
These solvers solve the (pure) monodomain equations on the form: find
the transmembrane potential :math:`v = v(x, t)` such that

.. math::

   v_t - \mathrm{div} ( G_i v) = I_s

where the subscript :math:`t` denotes the time derivative; :math:`G_i`
denotes a weighted gradient: :math:`G_i = M_i \mathrm{grad}(v)` for,
where :math:`M_i` is the intracellular cardiac conductivity tensor;
:math:`I_s` ise prescribed input. In addition, initial conditions are
given for :math:`v`:

.. math::

   v(x, 0) = v_0

Finally, boundary conditions must be prescribed. For now, this solver assumes pure homogeneous Neumann boundary conditions for :math:`v`.

"""

# Copyright (C) 2013 Johan Hake (hake@simula.no)
# Use and modify at will
# Last changed: 2013-04-18

__all__ = [
    "BasicMonodomainSolver",
    "MonodomainSolver"
]

from dolfin import *
from xalbrain.markerwisefield import *

from xalbrain.utils import (
    end_of_time,
    annotate_kwargs,
)

from typing import (
    Union,
    Dict,
    Tuple,
)


class BasicMonodomainSolver:
    """This solver is based on a theta-scheme discretization in time
    and CG_1 elements in space.

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

      I_s (:py:class:`dict`, optional)
        A typically time-dependent external stimulus given as a dict,
        with domain markers as the key and a
        :py:class:`dolfin.Expression` as values. NB: it is assumed
        that the time dependence of I_s is encoded via the 'time'
        Constant.

      v\_ (:py:class:`ufl.Expr`, optional)
        Initial condition for v. A new :py:class:`dolfin.Function`
        will be created if none is given.

      params (:py:class:`dolfin.Parameters`, optional)
        Solver parameters

    """
    def __init__(
            self,
            mesh: Mesh,
            time: Constant,
            M_i: Union[Expression, Dict[int, Expression]],
            I_s: Union[Expression, Dict[int, Expression]] = None,
            v_: Function = None,
            cell_domains = None,
            facet_domains = None,
            params: Parameters = None
    ) -> None:
        # Check some input
        assert isinstance(mesh, Mesh), \
            "Expecting mesh to be a Mesh instance, not %r" % mesh
        assert isinstance(time, Constant) or time is None, \
            "Expecting time to be a Constant instance (or None)."
        assert isinstance(params, Parameters) or params is None, \
            "Expecting params to be a Parameters instance (or None)"

        # Store input
        self._mesh = mesh
        self._I_s = I_s
        self._time = time

        # Initialize and update parameters if given
        self.parameters = self.default_parameters()
        if params is not None:
            self.parameters.update(params)

        # Set-up function spaces
        k = self.parameters["polynomial_degree"]
        V = FunctionSpace(self._mesh, "CG", k)

        self.V = V

        # Set-up solution fields:
        if v_ is None:
            self.v_ = Function(V, name="v_")
        else:
            # debug("Experimental: v_ shipped from elsewhere.")
            self.v_ = v_

        self.v = Function(self.V, name="v")

        if cell_domains is None:
            cell_domains = MeshFunction("size_t", mesh, self._mesh.geometry().dim())
            cell_domains.set_all(0)

        # Chech that it is indeed a cell function.
        cell_dim = cell_domains.dim()
        mesh_dim = self._mesh.geometry().dim()
        msg = "Got {}, expected {}.".format(cell_dim, mesh_dim)
        assert cell_dim == mesh_dim, msg
        self._cell_domains = cell_domains

        if facet_domains is None:
            facet_domains = MeshFunction("size_t", mesh, self._mesh.geometry().dim() - 1)
            facet_domains.set_all(0)

        # Check that it is indeed a facet function.
        facet_dim = facet_domains.dim()
        msg = "Got {}, expected {}.".format(facet_dim, mesh_dim)
        assert facet_dim == mesh_dim - 1, msg
        self._facet_domains = facet_domains

        if not isinstance(M_i, dict):
            M_i = {int(i): M_i for i in set(self._cell_domains.array())}
        else:
            M_i_keys = set(M_i.keys())
            cell_keys = set(self._cell_domains.array())
            msg = "Got {}, expected {}.".format(M_i_keys, cell_keys)
            assert M_i_keys == cell_keys, msg
        self._M_i = M_i

        # Figure out whether we should annotate or not
        self._annotate_kwargs = annotate_kwargs(self.parameters)

    @property
    def time(self) -> Constant:
        """The internal time of the solver."""
        return self._time

    def solution_fields(self) -> Tuple[Function]:
        """Return tuple of previous and current solution objects.

        Modifying these will modify the solution objects of the solver
        and thus provides a way for setting initial conditions for
        instance.

        *Returns*
          (previous v, current v) (:py:class:`tuple` of :py:class:`dolfin.Function`)
        """
        return self.v_, self.v

    def solve(self, interval, dt=None) -> None:
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
        # Initial set-up
        # Solve on entire interval if no interval is given.
        T0, T = interval
        if dt is None:
            dt = T - T0
        t0 = T0
        t1 = T0 + dt

        # Step through time steps until at end time
        while True:
            # info("Solving on t = (%g, %g)" % (t0, t1))
            self.step((t0, t1))

            # Yield solutions
            yield (t0, t1), self.solution_fields()

            # Break if this is the last step
            if end_of_time(T, t0, t1, dt):
                break

            # If not: update members and move to next time
            if isinstance(self.v_, Function):
                self.v_.assign(self.v)
            # else:
            #     debug("Assuming that v_ is updated elsewhere. Experimental.")

            t0 = t1
            t1 = t0 + dt

    def step(self, interval) -> None:
        r"""Solve on the given time interval (t0, t1).

        *Arguments*
          interval (:py:class:`tuple`)
            The time interval (t0, t1) for the step

        *Invariants*
          Assuming that v\_ is in the correct state for t0, gives
          self.v in correct state at t1.
        """
        # Extract interval and thus time-step
        t0, t1 = interval
        k_n = Constant(t1 - t0)

        # Extract theta parameter and conductivities
        theta = self.parameters["theta"]
        M_i = self._M_i

        # Set time
        t = t0 + theta*(t1 - t0)
        self.time.assign(t)

        # Get physical parameters
        chi = self.parameters["Chi"]
        capacitance = self.parameters["Cm"]
        lam = self.parameters["lambda"]
        lam_frac = Constant(lam/(1 + lam))

        # Define variational formulation
        v = TrialFunction(self.V)
        w = TestFunction(self.V)
        Dt_v_k_n = (v - self.v_)/k_n
        Dt_v_k_n *= chi*capacitance
        v_mid = theta*v + (1.0 - theta)*self.v_

        dz = Measure("dx", domain=self._mesh, subdomain_data=self._cell_domains)
        db = Measure("ds", domain=self._mesh, subdomain_data=self._facet_domains)
        # dz, rhs = rhs_with_markerwise_field(self._I_s, self._mesh, w)
        cell_tags = map(int, set(self._cell_domains.array()))   # np.int64 does not work
        facet_tags = map(int, set(self._facet_domains.array()))

        for key in cell_tags:
            G = Dt_v_k_n*w*dz(key)
            G += lam_frac*inner(M_i[key]*grad(v_mid), grad(w))*dz(key)

            if self._I_s is None:
                G -= chi*Constant(0)*w*dz(key)
            else:
                G -= chi*self._I_s*w*dz(key)

        # Define variational problem
        a, L = system(G)
        pde = LinearVariationalProblem(a, L, self.v)

        # Set-up solver
        solver_type = self.parameters["linear_solver_type"]
        solver = LinearVariationalSolver(pde)
        # solver.parameters.update(self.parameters["linear_variational_solver"])
        # solver.parameters["linear_solver"] = self.parameters["linear_solver_type"]
        solver.solve()

    @staticmethod
    def default_parameters():
        """
        Initialize and return a set of default parameters.

        *Returns*
          A set of parameters (:py:class:`dolfin.Parameters`)

        To inspect all the default parameters, do::

          info(BasicMonodomainSolver.default_parameters(), True)
        """
        params = Parameters("BasicMonodomainSolver")
        params.add("theta", 0.5)
        params.add("polynomial_degree", 1)
        params.add("enable_adjoint", False)
        params.add("default_timestep", 1.0)

        # Set default solver type to be iterative
        params.add("linear_solver_type", "direct")
        params.add("lu_type", "default")

        # Set default iterative solver choices (used if iterative
        # solver is invoked)
        params.add("algorithm", "cg")
        params.add("preconditioner", "petsc_amg")
        params.add("use_custom_preconditioner", False)

        params.add("Chi", 1.0)      # Membrane to volume ratio
        params.add("Cm", 1.0)      # Membrane Capacitance
        params.add("lambda", 1.0)

        # Add default parameters from both LU and Krylov solvers
        # params.add(LUSolver.default_parameters())
        # params.add(KrylovSolver.default_parameters())

        # # Customize default parameters for LUSolver
        # params["lu_solver"]["same_nonzero_pattern"] = True

        # params.add(LinearVariationalSolver.default_parameters())
        return params

class MonodomainSolver(BasicMonodomainSolver):
    __doc__ = BasicMonodomainSolver.__doc__

    def __init__(
            self,
            mesh: Mesh,
            time: Constant,
            M_i: Union[Expression, Dict[int, Expression]],
            I_s: Union[Expression, Dict[int, Expression]] = None,
            v_: Function = None,
            cell_domains: MeshFunction = None,
            facet_domains: MeshFunction = None,
            params: Parameters = None
    ) -> None:
        # Call super-class
        super().__init__(
            mesh,
            time,
            M_i,
            I_s=I_s,
            v_=v_,
            cell_domains=cell_domains,
            facet_domains=facet_domains,
            params=params)

        # Create variational forms
        self._timestep = Constant(self.parameters["default_timestep"])
        self._lhs, self._rhs, self._prec = self.variational_forms(self._timestep)

        # Preassemble left-hand side (will be updated if time-step changes)
        # debug("Preassembling monodomain matrix (and initializing vector)")
        self._lhs_matrix = assemble(self._lhs, **self._annotate_kwargs)
        self._rhs_vector = Vector(mesh.mpi_comm(), self._lhs_matrix.size(0))
        self._lhs_matrix.init_vector(self._rhs_vector, 0)

        # Create linear solver (based on parameter choices)
        self._linear_solver, self._update_solver = self._create_linear_solver()

    @property
    def linear_solver(self):
        """The linear solver (:py:class:`dolfin.LUSolver` or
        :py:class:`dolfin.KrylovSolver`)."""
        return self._linear_solver

    def _create_linear_solver(self):
        "Helper function for creating linear solver based on parameters."
        solver_type = self.parameters["linear_solver_type"]

        if solver_type == "direct":
            solver = LUSolver(self._lhs_matrix, self.parameters["lu_type"])
            # solver.parameters.update(self.parameters["lu_solver"])
            update_routine = self._update_lu_solver

        elif solver_type == "iterative":
            # Preassemble preconditioner (will be updated if time-step
            # changes)
            # debug("Preassembling preconditioner")
            # Initialize KrylovSolver with matrix and preconditioner
            alg = self.parameters["algorithm"]
            prec = self.parameters["preconditioner"]
            if self.parameters["use_custom_preconditioner"]:
                self._prec_matrix = assemble(self._prec,
                                             **self._annotate_kwargs)
                solver = PETScKrylovSolver(alg, prec)
                solver.parameters.update(self.parameters["krylov_solver"])
                solver.set_operators(self._lhs_matrix, self._prec_matrix)
                solver.ksp().setFromOptions()
            else:
                solver = PETScKrylovSolver(alg, prec)
                # solver.parameters.update(self.parameters["krylov_solver"])
                solver.set_operator(self._lhs_matrix)
                solver.ksp().setFromOptions()

            update_routine = self._update_krylov_solver
        else:
            assert False, "Unknown linear_solver_type given: %s"
            # error( % solver_type)
        return (solver, update_routine)

    @staticmethod
    def default_parameters():
        """Initialize and return a set of default parameters

        *Returns*
          A set of parameters (:py:class:`dolfin.Parameters`)

        To inspect all the default parameters, do::

          info(MonodomainSolver.default_parameters(), True)
        """
        params = Parameters("MonodomainSolver")
        params.add("enable_adjoint", False)
        params.add("theta", 0.5)
        params.add("polynomial_degree", 1)
        params.add("default_timestep", 1.0)

        # Set default solver type to be iterative
        params.add("linear_solver_type", "direct")
        params.add("lu_type", "default")

        # Set default iterative solver choices (used if iterative
        # solver is invoked)
        params.add("algorithm", "cg")
        params.add("preconditioner", "petsc_amg")
        params.add("use_custom_preconditioner", False)


        params.add("Chi", 1.0)        # Membrane to volume ratio
        params.add("Cm", 1.0)         # Membrane capacitance
        params.add("lambda", 1.0)

        # # Add default parameters from both LU and Krylov solvers
        # params.add(LUSolver.default_parameters())
        # params.add(KrylovSolver.default_parameters())

        # Customize default parameters for LUSolver
        # params["lu_solver"]["same_nonzero_pattern"] = True

        # Customize default parameters for KrylovSolver
        #params["krylov_solver"]["preconditioner"]["structure"] = "same"
        return params

    def variational_forms(self, k_n: Constant):
        """Create the variational forms corresponding to the given
        discretization of the given system of equations.

        *Arguments*
          k_n (:py:class:`ufl.Expr` or float)
            The time step

        *Returns*
          (lhs, rhs, prec) (:py:class:`tuple` of :py:class:`ufl.Form`)
        """
        # Extract theta parameter and conductivities
        theta = self.parameters["theta"]
        M_i = self._M_i

        # Define variational formulation
        v = TrialFunction(self.V)
        w = TestFunction(self.V)

        chi = self.parameters["Chi"]
        capacitance = self.parameters["Cm"]
        lam = self.parameters["lambda"]
        lam_frac = Constant(lam/(1 + lam))

        # Set-up variational problem
        Dt_v_k_n = (v - self.v_)/k_n
        Dt_v_k_n *= chi*capacitance
        v_mid = theta*v + (1.0 - theta)*self.v_

        # dz, rhs = rhs_with_markerwise_field(self._I_s, self._mesh, w)

        dz = Measure("dx", domain=self._mesh, subdomain_data=self._cell_domains)
        db = Measure("ds", domain=self._mesh, subdomain_data=self._facet_domains)
        cell_tags = map(int, set(self._cell_domains.array()))   # np.int64 does not work
        facet_tags = map(int, set(self._facet_domains.array()))

        prec = 0

        for key in cell_tags:
            G = Dt_v_k_n*w*dz(key)
            G += lam_frac*inner(M_i[key]*grad(v_mid), grad(w))*dz(key)

            if self._I_s is None:
                G -= chi*Constant(0)*w*dz(key)
            else:
                G -= chi*self._I_s*w*dz(key)


        # Define preconditioner based on educated(?) guess by Marie
        prec += (v*w + k_n/2.0*inner(M_i[key]*grad(v), grad(w)))*dz(key)

        a, L = system(G)
        return (a, L, prec)

    def step(self, interval):
        """Solve on the given time step (t0, t1).

        *Arguments*
          interval (:py:class:`tuple`)
            The time interval (t0, t1) for the step

        *Invariants*
          Assuming that v\_ is in the correct state for t0, gives
          self.v in correct state at t1.
        """
        timer = Timer("PDE Step")

        # Extract interval and thus time-step
        t0, t1 = interval
        dt = t1 - t0
        theta = self.parameters["theta"]
        t = t0 + theta*dt
        self.time.assign(t)

        # Update matrix and linear solvers etc as needed
        timestep_unchanged = (abs(dt - float(self._timestep)) < 1.e-12)
        self._update_solver(timestep_unchanged, dt)

        # Assemble right-hand-side
        timer0 = Timer("Assemble rhs")
        assemble(self._rhs, tensor=self._rhs_vector, **self._annotate_kwargs)
        del timer0

        # Solve problem
        self.linear_solver.solve(
            self.v.vector(),
            self._rhs_vector,
            **self._annotate_kwargs
        )
        timer.stop()

    def _update_lu_solver(self, timestep_unchanged, dt):
        """Helper function for updating an LUSolver depending on
        whether timestep has changed."""

        # Update reuse of factorization parameter in accordance with
        # changes in timestep
        if timestep_unchanged:
            # debug("Timestep is unchanged, reusing LU factorization")
            # self.linear_solver.parameters["reuse_factorization"] = True
            pass
        else:
            # debug("Timestep has changed, updating LU factorization")
            # self.linear_solver.parameters["reuse_factorization"] = False

            # Update stored timestep
            # FIXME: dolfin_adjoint still can't annotate constant assignment.
            self._timestep.assign(Constant(dt))#, annotate=annotate)

            # Reassemble matrix
            assemble(self._lhs, tensor=self._lhs_matrix, **self._annotate_kwargs)

    def _update_krylov_solver(self, timestep_unchanged, dt):
        """Helper function for updating a KrylovSolver depending on
        whether timestep has changed."""

        kwargs = annotate_kwargs(self.parameters)
        # Update reuse of preconditioner parameter in accordance with
        # changes in timestep
        if timestep_unchanged:
            # debug("Timestep is unchanged, reusing preconditioner")
            #self.linear_solver.parameters["preconditioner"]["structure"] = "same"
            pass
        else:
            # debug("Timestep has changed, updating preconditioner")
            #self.linear_solver.parameters["preconditioner"]["structure"] = \
            #                                            "same_nonzero_pattern"

            # Update stored timestep
            self._timestep.assign(Constant(dt))

            # Reassemble matrix
            assemble(self._lhs, tensor=self._lhs_matrix, **self._annotate_kwargs)

            # Reassemble preconditioner
            if self.parameters["use_custom_preconditioner"]:
                assemble(self._prec, tensor=self._prec_matrix, **self._annotate_kwargs)

        # Set nonzero initial guess if it indeed is nonzero
        # if (self.v.vector().norm("l2") > 1.e-12):
        #     # debug("Initial guess is non-zero.")
        #     self.linear_solver.parameters["nonzero_initial_guess"] = True
