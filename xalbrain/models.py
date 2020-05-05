"""This module contains a container class for cardiac models:
:py:class:`~xalbrain.cardiacmodels.Model`.  This class
should be instantiated for setting up specific cardiac simulation
scenarios.
"""

# Copyright (C) 2012 Marie E. Rognes (meg@simula.no)
# Use and modify at will
# Last changed: 2016-04-21

import typing as tp

import dolfin as df

import xalbrain as xb

from .cellmodels import CellModel


__all__ = ["Model"] 


class Model:
    """
    A container class for cardiac models. Objects of this class
    represent a specific cardiac simulation set-up and should provide

    * A computational domain
    * A cardiac cell model
    * Intra-cellular and extra-cellular conductivities
    * Various forms of stimulus (optional).

    This container class is designed for use with the splitting
    solvers (:py:mod:`xalbrain.splittingsolver`), see their
    documentation for more information on how the attributes are
    interpreted in that context.

    *Arguments*
      domain (:py:class:`dolfin.Mesh`)
        the computational domain in space
      time (:py:class:`dolfin.Constant` or None )
        A constant holding the current time.
      M_i (:py:class:`ufl.Expr`)
        the intra-cellular conductivity as an ufl Expression
      M_e (:py:class:`ufl.Expr`)
        the extra-cellular conductivity as an ufl Expression
      cell_models (:py:class:`~xalbrain.cellmodels.cardiaccellmodel.CellModel`)
        a cell model or a dict with cell models associated with a cell model domain
      stimulus (:py:class:`dict`, optional)
        A typically time-dependent external stimulus given as a dict,
        with domain markers as the key and a
        :py:class:`dolfin.Expression` as values. NB: it is assumed
        that the time dependence of I_s is encoded via the 'time'
        Constant.
      applied_current (:py:class:`ufl.Expr`, optional)
        an applied current as an ufl Expression

    """

    def __init__(
            self,
            domain: df.Mesh,
            time: df.Constant,
            M_i: tp.Union[df.Expression, tp.Dict[int, df.Expression]],
            M_e: tp.Union[df.Expression, tp.Dict[int, df.Expression]],
            cell_models: CellModel,
            stimulus: tp.Union[df.Expression, tp.Dict[int, df.Expression]] = None,
            applied_current: tp.Union[df.Expression, tp.Dict[int, df.Expression]] = None,
            ect_current: tp.Dict[int, df.Expression] = None,
            dirichlet_bc_u: tp.List[tp.Tuple[df.Expression, int]] = None,
            dirichlet_bc_v: tp.List[tp.Tuple[df.Expression, int]] = None,
            cell_domains: df.MeshFunction = None,
            facet_domains: df.MeshFunction = None,
            indicator_function: df.Function = None
    ) -> None:
        """Create Model from given input."""
        self._ect_current = ect_current

        # Check input and store attributes
        msg = "Expecting domain to be a Mesh instance, not %r" % domain
        assert isinstance(domain, df.Mesh), msg
        self._domain = domain

        msg = "Expecting time to be a Constant instance, not %r." % time
        assert isinstance(time, df.Constant) or time is None, msg
        self._time = time

        self._intracellular_conductivity = M_i
        self._extracellular_conductivity = M_e

        self._cell_domains = cell_domains
        self._facet_domains = facet_domains

        # indicator function
        self._indicator_functionn = indicator_function

        # Handle cell_models
        self._cell_models = cell_models

        # Handle stimulus
        self._stimulus = stimulus

        # Handle applied current
        self._applied_current = applied_current
        self._dirichlet_bcs_u = dirichlet_bc_u
        self._dirichlet_bcs_v = dirichlet_bc_v

    @property
    def dirichlet_bc_u(self) -> tp.List[tp.Tuple[df.Expression, int]]:
        """Return a list of `DirichletBC`s u."""
        return self._dirichlet_bcs_v

    @property
    def dirichlet_bc_v(self) -> tp.List[tp.Tuple[df.Expression, int]]:
        """Return a lit of `DirichletBC`'s for v."""
        return self._dirichlet_bcs_u

    @property
    def ect_current(self) -> df.Expression:
        """Return the neumnn current."""
        return self._ect_current

    @property
    def applied_current(self) -> tp.Any:
        "An applied current: used as a source in the elliptic bidomain equation"
        return self._applied_current

    @property
    def stimulus(self) -> tp.Any:
        "A stimulus: used as a source in the parabolic bidomain equation"
        return self._stimulus

    def conductivities(self) -> tp.Any:
        """Return the intracellular and extracellular conductivities
        as a tuple of UFL Expressions.

        *Returns*
        (M_i, M_e) (:py:class:`tuple` of :py:class:`ufl.Expr`)
        """
        return self.intracellular_conductivity(), self.extracellular_conductivity()

    def intracellular_conductivity(self) -> tp.Any:
        """The intracellular conductivity (:py:class:`ufl.Expr`)."""
        return self._intracellular_conductivity

    def extracellular_conductivity(self) -> tp.Any:
        """The intracellular conductivity (:py:class:`ufl.Expr`)."""
        return self._extracellular_conductivity

    @property
    def time(self) -> df.Constant:
        """The current time (:py:class:`dolfin.Constant` or None)."""
        return self._time

    @property
    def mesh(self) -> df.Mesh:
        """The spatial domain (:py:class:`dolfin.Mesh`)."""
        return self._domain

    @property
    def cell_domains(self) -> df.MeshFunction:
        """Marked volume."""
        return self._cell_domains

    @property
    def facet_domains(self) -> df.MeshFunction:
        """Marked area."""
        return self._facet_domains

    @property
    def cell_models(self) -> xb.cellmodels.CellModel:
        """Return the cell models."""
        return self._cell_models

    @property
    def indicator_function(self) -> df.Function:
        """Return indicator function."""
        return self._indicator_functionn
