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

from xalbrain.dolfinimport import *
from xalbrain.markerwisefield import *

from xalbrain.utils import (
    end_of_time,
    annotate_kwargs,
)

from typing import (
    Union,
    Dict,
    Tuple,
    Generator
)

from xalbrain.parameters import (
    MonodomainParameters,
    KrylovParmeters,
    LUParameters,
)

import ufl


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
            parameters: MonodomainParameters,
            linear_solver_parameters: Union[KrylovParmeters, LUParameters],
            I_s: Union[Expression, Dict[int, Expression]]=None,
            v_: Function=None,
    ) -> None:
        # Store input
        self._mesh = mesh
        self._M_i = M_i
        self._time = time
        self._I_s = I_s

        # Set-up function spaces
        k = self.parameters.polynomial_degree
        V = FunctionSpace(self._mesh, "CG", k)

        self.V = V

        # Set-up solution fields:
        if v_ is None:
            self.v_ = Function(V, name="v_")
        else:
            debug("Experimental: v_ shipped from elsewhere.")
            self.v_ = v_

        self.v = Function(self.V, name="v")

        # Figure out whether we should annotate or not
        # self._annotate_kwargs = annotate_kwargs(self.parameters)

    @property
    def time(self) -> Constant:
        """The internal time of the solver."""
        return self._time

    def solution_fields(self) -> Tuple[Function, Function]:
        """Return tuple of previous and current solution objects.

        Modifying these will modify the solution objects of the solver
        and thus provides a way for setting initial conditions for
        instance.

        Returns:
            (previous v, current v)
        """
        return self.v_, self.v

    def solve(
            self,
            interval: Tuple[float, float],
            dt: float
    ) -> Generator[Tuple[Tuple[float, float], Function], None, None]:
        """Solve the problem in the interval with the specified time step.

        Arguments:
            interval: The time interval for the solve given by (t0, t1).
            dt: The timestep for the solve. Defaults to length of interval.

        Returns:
            (timestep, solution_field)

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
        t0 = T0
        t1 = T0 + dt

        # Step through time steps until at end time
        while True:
            info("Solving on t = ({:g}, {:g})".format(t0, t1))
            self._step((t0, t1))

            # Yield solutions
            yield (t0, t1), self.solution_fields()

            # Break if this is the last step
            if end_of_time(T, t0, t1, dt):
                break

            # If not: update members and move to next time
            if isinstance(self.v_, Function):
                self.v_.assign(self.v)
            else:
                debug("Assuming that v_ is updated elsewhere. Experimental.")

            t0 = t1
            t1 = t0 + dt

    def _step(self, interval: Tuple[float, float]) -> None:
        r"""Solve on the given time interval (t0, t1).

        Arguments:
            interval: The time interval (t0, t1) for the step

        Invariants:
            Assuming that v\_ is in the correct state for t0, gives
            self.v in correct state at t1.
        """
        # Extract interval and thus time-step
        t0, t1 = interval
        k_n = Constant(t1 - t0)

        # Extract theta parameter and conductivities
        theta = self.parameters.theta
        M_i = self._M_i

        # Set time
        t = t0 + theta*(t1 - t0)
        self.time.assign(t)

        # Define variational formulation
        v = TrialFunction(self.V)
        w = TestFunction(self.V)
        Dt_v_k_n = (v - self.v_)/k_n
        v_mid = theta*v + (1.0 - theta)*self.v_

        dz, rhs = rhs_with_markerwise_field(self._I_s, self._mesh, w)
        G = Dt_v_k_n*w*dz()
        G += inner(M_i*grad(v_mid), grad(w))*dz()
        G -= rhs

        # Define variational problem
        a, L = system(G)
        pde = LinearVariationalProblem(a, L, self.v)

        # Set-up solver
        solver = LinearVariationalSolver(pde)
        # FIXME: Figure out what to do about LU vs Krylov
        # solver.parameters.update(LinearVariationalProblem.default_parameters())
        # solver.parameters["linear_solver"] = self.parameters.solver_type 
        solver.solve()

    # @staticmethod
    # def default_parameters():
    #     """
    #     Initialize and return a set of default parameters.

    #     *Returns*
    #       A set of parameters (:py:class:`dolfin.Parameters`)

    #     To inspect all the default parameters, do::

    #       info(BasicMonodomainSolver.default_parameters(), True)
    #     """
    #     params = Parameters("BasicMonodomainSolver")
    #     params.add("theta", 0.5)
    #     params.add("polynomial_degree", 1)
    #     params.add("enable_adjoint", False)
    #     params.add("default_timestep", 1.0)     # FIXME: What is this

    #     # Set default solver type to be iterative
    #     params.add("linear_solver_type", "direct")
    #     params.add("lu_type", "default")        # FIXME: What is this

    #     # Set default iterative solver choices (used if iterative
    #     # solver is invoked)
    #     params.add("algorithm", "cg")
    #     params.add("preconditioner", "petsc_amg")
    #     params.add("use_custom_preconditioner", False)

    #     # Add default parameters from both LU and Krylov solvers
    #     params.add(LUSolver.default_parameters())
    #     params.add(KrylovSolver.default_parameters())

    #     # Customize default parameters for LUSolver
    #     params["lu_solver"]["same_nonzero_pattern"] = True

    #     params.add(LinearVariationalSolver.default_parameters())
    #     return params


class MonodomainSolver(BasicMonodomainSolver):
    __doc__ = BasicMonodomainSolver.__doc__

    def __init__(
            self,
            mesh: Mesh,
            time: Constant,
            M_i: Union[Expression, Dict[int, Expression]],
            parameter: MonodomainParameters,
            linear_solver_parameters: Union[KrylovParmeters, LUParameters],
            I_s: Union[Expression, Dict[int, Expression]]=None,
            v_: Function=None,
    ) -> None:
        super().__init__(
            mesh, time, M_i, parameters, linear_solver_parameters, I_s, v)
        )

        # Create variational forms
        self._timestep = Constant(self.parameters["default_timestep"])
        self._lhs, self._rhs, self._prec = self.variational_forms(self._timestep)

        # Preassemble left-hand side (will be updated if time-step changes)
        debug("Preassembling monodomain matrix (and initializing vector)")
        self._lhs_matrix = assemble(self._lhs, **self._annotate_kwargs)
        self._rhs_vector = Vector(mesh.mpi_comm(), self._lhs_matrix.size(0))
        self._lhs_matrix.init_vector(self._rhs_vector, 0)

        # Create linear solver (based on parameter choices)
        self._linear_solver, self._update_solver = self._create_linear_solver()

    @property
    def linear_solver(self) -> Union[KrylovSolver, LUSolver]:
        """The linear solver (:py:class:`dolfin.LUSolver` or
        :py:class:`dolfin.KrylovSolver`)."""
        return self._linear_solver

    def _create_linear_solver(self):
        "Helper function for creating linear solver based on parameters."
        solver_type = self.parameters.linear_solver_type

        if solver_type == "direct":
            _sp = LUSolver.default_parameters()
            _sp["linear_solver"] = self.linear_solver_parameters.solver
            solver = LUSolver(
                self._lhs_matrix,
                _sp
            )
            update_routine = self._update_lu_solver

        elif solver_type == "iterative":
            # Preassemble preconditioner (will be updated if time-step
            # changes)
            debug("Preassembling preconditioner")
            # Initialize KrylovSolver with matrix and preconditioner
            alg = self.parameters.solver
            prec = self.parameters.preconditioner
            # if self.parameters["use_custom_preconditioner"]:
            #     self._prec_matrix = assemble(self._prec,
            #                                  **self._annotate_kwargs)
            #     solver = PETScKrylovSolver(alg, prec)
            #     solver.parameters.update(self.parameters["krylov_solver"])
            #     solver.set_operators(self._lhs_matrix, self._prec_matrix)
            #     solver.ksp().setFromOptions()
            # else:
            solver = PETScKrylovSolver(alg, prec)
            _sp = PETScKrylovSolver.default_parameters()
            _sp["absolute_tolerance"] = self.parameters.absolute_tolerance
            _sp["relative_tolerance"] = self.parameters.relative_tolerance
            _sp["nonzero_initial_guess"] =  self.parameters.nonzero_initial_guess
            solver.parameters.update(_sp)
            solver.set_operator(self._lhs_matrix)
            solver.ksp().setFromOptions()
            update_routine = self._update_krylov_solver
        else:
            error("Unknown linear_solver_type given: {}".format(solver_type))
        return solver, update_routine

    #@staticmethod
    #def default_parameters():
    #    """Initialize and return a set of default parameters

    #    *Returns*
    #      A set of parameters (:py:class:`dolfin.Parameters`)

    #    To inspect all the default parameters, do::

    #      info(MonodomainSolver.default_parameters(), True)
    #    """
    #    params = Parameters("MonodomainSolver")
    #    params.add("enable_adjoint", False)
    #    params.add("theta", 0.5)
    #    params.add("polynomial_degree", 1)
    #    params.add("default_timestep", 1.0)

    #    # Set default solver type to be iterative
    #    params.add("linear_solver_type", "direct")
    #    params.add("lu_type", "default")

    #    # Set default iterative solver choices (used if iterative
    #    # solver is invoked)
    #    params.add("algorithm", "cg")
    #    params.add("preconditioner", "petsc_amg")
    #    params.add("use_custom_preconditioner", False)

    #    # Add default parameters from both LU and Krylov solvers
    #    params.add(LUSolver.default_parameters())
    #    params.add(KrylovSolver.default_parameters())

    #    # Customize default parameters for LUSolver
    #    params["lu_solver"]["same_nonzero_pattern"] = True

    #    # Customize default parameters for KrylovSolver
    #    #params["krylov_solver"]["preconditioner"]["structure"] = "same"
    #    return params

    def variational_forms(self, k_n: Constant) -> Tuple[ufl.Form, ufl.Form. ufl.Form]:
        """Create the variational forms corresponding to the given
        discretization of the given system of equations.

        Arguments:
            k_n: The time step.

        Returns:
            (lhs, rhs, prec)
        """
        # Extract theta parameter and conductivities
        theta = self.parameters.theta
        M_i = self._M_i

        # Define variational formulation
        v = TrialFunction(self.V)
        w = TestFunction(self.V)

        # Set-up variational problem
        Dt_v_k_n = (v - self.v_)/k_n
        v_mid = theta*v + (1.0 - theta)*self.v_

        dz, rhs = rhs_with_markerwise_field(self._I_s, self._mesh, w)
        G = Dt_v_k_n*w*dz()
        G += inner(M_i*grad(v_mid), grad(w))*dz()
        G -= rhs

        # Define preconditioner based on educated(?) guess by Marie
        prec = (v*w + k_n/2.0*inner(M_i*grad(v), grad(w)))*dz

        a, L = system(G)
        return a, L, prec

    def _step(self, interval: Tuple[float, float]) -> None:
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
        theta = self.parameters.theta
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

    def _update_lu_solver(self, timestep_unchanged: bool, dt: float) -> None:
        """Helper function for updating an LUSolver depending on
        whether timestep has changed."""
        # Update reuse of factorization parameter in accordance with
        # changes in timestep
        if timestep_unchanged:
            debug("Timestep is unchanged, reusing LU factorization")
            self.linear_solver.parameters["reuse_factorization"] = True
        else:
            debug("Timestep has changed, updating LU factorization")
            self.linear_solver.parameters["reuse_factorization"] = False

            # Update stored timestep
            # FIXME: dolfin_adjoint still can't annotate constant assignment.
            self._timestep.assign(Constant(dt))#, annotate=annotate)

            # Reassemble matrix
            assemble(self._lhs, tensor=self._lhs_matrix, **self._annotate_kwargs)

    def _update_krylov_solver(self, timestep_unchanged: bool, dt: float):
        """Helper function for updating a KrylovSolver depending on
        whether timestep has changed."""
        kwargs = annotate_kwargs(self.parameters)
        # Update reuse of preconditioner parameter in accordance with
        # changes in timestep
        if timestep_unchanged:
            debug("Timestep is unchanged, reusing preconditioner")
            #self.linear_solver.parameters["preconditioner"]["structure"] = "same"
        else:
            debug("Timestep has changed, updating preconditioner")
            #self.linear_solver.parameters["preconditioner"]["structure"] = \
            #                                            "same_nonzero_pattern"

            # Update stored timestep
            self._timestep.assign(Constant(dt))

            # Reassemble matrix
            assemble(self._lhs, tensor=self._lhs_matrix, **self._annotate_kwargs)

            # Reassemble preconditioner
            # if self.parameters["use_custom_preconditioner"]:
            #     assemble(self._prec, tensor=self._prec_matrix, **self._annotate_kwargs)

        # Set nonzero initial guess if it indeed is nonzero
        if (self.v.vector().norm("l2") > 1.e-12):
            debug("Initial guess is non-zero.")
            self.linear_solver.parameters["nonzero_initial_guess"] = True
