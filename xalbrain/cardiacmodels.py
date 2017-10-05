"""This module contains a container class for cardiac models:
:py:class:`~xalbrain.cardiacmodels.CardiacModel`.  This class
should be instantiated for setting up specific cardiac simulation
scenarios.
"""

# Copyright (C) 2012 Marie E. Rognes (meg@simula.no)
# Use and modify at will
# Last changed: 2016-04-21

__all__ = ["CardiacModel"]

from xalbrain.dolfinimport import (
    Parameters,
    Mesh,
    Constant,
    GenericFunction,
    error,
)

from xalbrain.markerwisefield import (
    Markerwise,
    handle_markerwise,
)
from .cellmodels import *

# ------------------------------------------------------------------------------
# Cardiac models
# ------------------------------------------------------------------------------

class CardiacModel(object):
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
    def __init__(self, domain, time, M_i, M_e, cell_models,
                 stimulus=None, applied_current=None,
                 cell_domains=None, facet_domains=None):
        "Create CardiacModel from given input."

        self._handle_input(domain, time, M_i, M_e, cell_models,
                           stimulus, applied_current,
                           facet_domains, cell_domains)

    def _handle_input(self, domain, time, M_i, M_e, cell_models,
                      stimulus=None, applied_current=None,
                      facet_domains=None, cell_domains=None):

        # Check input and store attributes
        msg = "Expecting domain to be a Mesh instance, not %r" % domain
        assert isinstance(domain, Mesh), msg
        self._domain = domain

        msg = "Expecting time to be a Constant instance, not %r." % time
        assert isinstance(time, Constant) or time is None, msg
        self._time = time

        self._intracellular_conductivity = M_i
        self._extracellular_conductivity = M_e

        self._cell_domains = cell_domains
        self._facet_domains = facet_domains

        # Handle cell_models
        self._cell_models = handle_markerwise(cell_models, CardiacCellModel)
        if isinstance(self._cell_models, Markerwise):
            msg = "Different cell_models are currently not supported."
            error(msg)

        # Handle stimulus
        self._stimulus = handle_markerwise(stimulus, GenericFunction)

        # Handle applied current
        ac = applied_current
        self._applied_current = handle_markerwise(ac, GenericFunction)

    def applied_current(self):
        "An applied current: used as a source in the elliptic bidomain equation"
        return self._applied_current

    def stimulus(self):
        "A stimulus: used as a source in the parabolic bidomain equation"
        return self._stimulus

    def conductivities(self):
        """Return the intracellular and extracellular conductivities
        as a tuple of UFL Expressions.

        *Returns*
        (M_i, M_e) (:py:class:`tuple` of :py:class:`ufl.Expr`)
        """
        return (self.intracellular_conductivity(),
                self.extracellular_conductivity())

    def intracellular_conductivity(self):
        "The intracellular conductivity (:py:class:`ufl.Expr`)."
        return self._intracellular_conductivity

    def extracellular_conductivity(self):
        "The intracellular conductivity (:py:class:`ufl.Expr`)."
        return self._extracellular_conductivity

    def time(self):
        "The current time (:py:class:`dolfin.Constant` or None)."
        return self._time

    def domain(self):
        "The spatial domain (:py:class:`dolfin.Mesh`)."
        return self._domain

    def cell_domains(self):
        "Marked volume"
        return self._cell_domains

    def facet_domains(self):
        "Marked area"
        return self._facet_domains

    def cell_models(self):
        "Return the cell models"
        return self._cell_models
