"""This module contains a container class for cardiac models:
:py:class:`~xalbrain.cardiacmodels.CardiacModel`.  This class
should be instantiated for setting up specific cardiac simulation
scenarios.
"""

# Copyright (C) 2012 Marie E. Rognes (meg@simula.no)
# Use and modify at will
# Last changed: 2016-04-21

import dolfin as df

import xalbrain as xb

from xalbrain.markerwisefield import (
    Markerwise,
)

from .cellmodels import *       # Why do I need this?

from typing import (
    Dict,
    Union,
    List,
    Tuple,
    Any,
)


__all__ = ["CardiacModel"] 


class CardiacModel:
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
      cell_models (:py:class:`~xalbrain.cellmodels.cardiaccellmodel.CardiacCellModel`)
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
            M_i: Union[df.Expression, Dict[int, df.Expression]],
            M_e: Union[df.Expression, Dict[int, df.Expression]],
            cell_models: CardiacCellModel,
            stimulus: Union[df.Expression, Dict[int, df.Expression]] = None,
            applied_current: Union[df.Expression, Dict[int, df.Expression]] = None,
            ect_current: Dict[int, df.Expression] = None,
            dirichlet_bc_u: List[Tuple[df.Expression, int]] = None,
            dirichlet_bc_v: List[Tuple[df.Expression, int]] = None,
            cell_domains: df.MeshFunction = None,
            facet_domains: df.MeshFunction = None
    ) -> None:
        """Create CardiacModel from given input."""
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

        # Handle cell_models
        self._cell_models = cell_models
        if isinstance(self._cell_models, Markerwise):
            msg = "Different cell_models are currently not supported."
            df.error(msg)

        # Handle stimulus
        self._stimulus = stimulus

        # Handle applied current
        self._applied_current = applied_current
        self._dirichlet_bcs_u = dirichlet_bc_u
        self._dirichlet_bcs_v = dirichlet_bc_v

    @property
    def dirichlet_bc_u(self) -> List[Tuple[df.Expression, int]]:
        """Return a list of `DirichletBC`s u."""
        return self._dirichlet_bcs_v

    @property
    def dirichlet_bc_v(self) -> List[Tuple[df.Expression, int]]:
        """Return a lit of `DirichletBC`'s for v."""
        return self._dirichlet_bcs_u

    @property
    def ect_current(self) -> df.Expression:
        """Return the neumnn current."""
        return self._ect_current

    def applied_current(self) -> Any:
        "An applied current: used as a source in the elliptic bidomain equation"
        return self._applied_current

    def stimulus(self) -> Any:
        "A stimulus: used as a source in the parabolic bidomain equation"
        return self._stimulus

    def conductivities(self) -> Any:
        """Return the intracellular and extracellular conductivities
        as a tuple of UFL Expressions.

        *Returns*
        (M_i, M_e) (:py:class:`tuple` of :py:class:`ufl.Expr`)
        """
        return self.intracellular_conductivity(), self.extracellular_conductivity()

    def intracellular_conductivity(self) -> Any:
        """The intracellular conductivity (:py:class:`ufl.Expr`)."""
        return self._intracellular_conductivity

    def extracellular_conductivity(self) -> Any:
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
    def cell_models(self) -> xb.cellmodels.CardiacCellModel:
        """Return the cell models."""
        return self._cell_models
