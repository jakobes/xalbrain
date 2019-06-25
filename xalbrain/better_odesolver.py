import time

import numpy as np
import dolfin as df

from xalode import (
    VectorDouble,
    VectorInt,
)

from extension_modules import load_module
from xalbrain.cellsolver import BasicCardiacODESolver

from xalbrain.utils import TimeStepper

from typing import (
    Tuple,
    Union,
    Dict,
)

from xalbrain.cellmodels import CardiacCellModel


class BetterODESolver(BasicCardiacODESolver):

    def __init__(
            self,
            mesh: df.Mesh,
            time: df.Constant,
            model: CardiacCellModel,
            I_s: Union[df.Expression, Dict[int, df.Expression]] = None,     # FIXME: for compatibility?
            cell_domains = None,
            valid_cell_tags = None,
            reload_ext_modules: bool = False,
            params: df.Parameters = None
    ) -> None:
        """Initialise parameters. NB! Keep I_s for compatibility"""
        # Store input
        self._mesh = mesh
        self._time = time
        self._model = model     # FIXME: For initial conditions

        # Extract some information from cell model
        self._num_states = self._model.num_states()

        # Create time if not given, otherwise use given time
        if time is None:
            self._time = df.Constant(0.0)
        else:
            self._time = time

        # Initialize and update parameters if given
        self.parameters = self.default_parameters()
        if params is not None:
            self.parameters.update(params)

        # Create (vector) function space for potential + states
        self.VS = df.VectorFunctionSpace(self._mesh, "CG", 1, dim=self._num_states + 1)

        # Initialize solution field
        self.vs_ = df.Function(self.VS, name="vs_")
        self.vs = df.Function(self.VS, name="vs")

        def _masked_dofs(dofmap, cell_domains_array, valid_cell_tags):
            mask_list = []
            for i, ct in enumerate(cell_domains_array):
                cell_dofs = dofmap.cell_dofs(i)
                if ct in valid_cell_tags:
                    mask_list += list(cell_dofs)
            foo = np.unique(mask_list)
            return VectorInt(np.unique(mask_list))

        dofmaps = [
            self.VS.sub(i).dofmap() for i in range(self.VS.num_sub_spaces())
        ]


        if len(valid_cell_tags) == 0:
            self.dofs = [VectorInt(dofmap.dofs()) for dofmap in dofmaps]
        else:
            self.dofs = [
                _masked_dofs(dofmap, cell_domains.array(), valid_cell_tags) for dofmap in dofmaps
            ]

        # from IPython import embed; embed()
        # 1/0

        model_name = model.__class__.__name__
        self.ode_module = load_module(
            model_name,
            recompile=reload_ext_modules,
            verbose=reload_ext_modules
        )

        self.ode_solver = self.ode_module.LatticeODESolver(*self.dofs)

    @staticmethod
    def default_parameters():
        params = df.Parameters("BetterODESolver")
        params.add("dt_fraction", 1)
        return params

    @property
    def time(self) -> df.Constant:
        """The internal time of the solver."""
        return self._time

    def solution_fields(self) -> Tuple[df.Function, df.Function]:
        """
        Return current solution object.

        Modifying this will modify the solution object of the solver
        and thus provides a way for setting initial conditions for
        instance.

        *Returns*
          (previous vs_, current vs) (:py:class:`dolfin.Function`)
        """
        return self.vs_, self.vs

    def step(self, t0: float, t1: float) -> None:
        """Take a step using my much better ode solver."""
        # t0, t1 = interval
        dt = t1 - t0        # TODO: Is this risky?

        self.ode_solver.solve(self.vs_.vector(), t0, t1, dt)
        self.vs.assign(self.vs_)

    def solve(self, interval: Tuple[float, float], dt: float = None, verbose: bool = False):
        """
        Solve the problem given by the model on a given time interval
        (t0, t1) with a given timestep dt and return generator for a
        tuple of the interval and the current vs solution.

        *Arguments*
          interval (:py:class:`tuple`)
            The time interval for the solve given by (t0, t1)
          dt (int, optional)
            The timestep for the solve. Defaults to length of interval

        *Returns*
          (timestep, current vs) via (:py:class:`genexpr`)

        *Example of usage*::

          # Create generator
          solutions = solver.solve((0.0, 1.0), 0.1)

          # Iterate over generator (computes solutions as you go)
          for (interval, vs) in solutions:
            # do something with the solutions

        """
        # Initial time set-up
        T0, T = interval

        # Solve on entire interval if no interval is given.
        if dt is None:
            dt = T - T0

        # Create timestepper
        time_stepper = TimeStepper(interval, dt)

        for t0, t1 in time_stepper:
            # df.info_blue("Solving on t = (%g, %g)" % (t0, t1))
            tick = time.perf_counter()
            self.step((t0, t1))
            tock = time.perf_counter()
            if verbose:
                print("ODE time: ", tock - tick)

            # Yield solutions
            yield (t0, t1), self.vs
            self.vs_.assign(self.vs)


class BetterSingleCellSolver(BetterODESolver):
    def __init__(
            self,
            model: CardiacCellModel,
            time: df.Constant,
            reload_ext_modules: bool = False,
            params: df.Parameters = None
    ) -> None:
        """Create solver from given cell model and optional parameters."""
        assert isinstance(model, CardiacCellModel), \
            "Expecting model to be a CardiacCellModel, not %r" % model
        assert (isinstance(time, df.Constant)), \
            "Expecting time to be a Constant instance, not %r" % time
        assert isinstance(params, df.Parameters) or params is None, \
            "Expecting params to be a Parameters (or None), not %r" % params

        # Store model
        self._model = model

        # Define carefully chosen dummy mesh
        mesh = df.UnitIntervalMesh(1)

        super().__init__(
            mesh,
            time,
            model,
            I_s=model.stimulus,
            reload_ext_modules=reload_ext_modules,
            params=params
        )
