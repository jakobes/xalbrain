"""This module contains a container class for cardiac models:
:py:class:`~beatadjoint.cardiacmodels.CardiacModel`.  This class
should be instantiated for setting up specific cardiac simulation
scenarios.
"""

# Copyright (C) 2012 Marie E. Rognes (meg@simula.no)
# Use and modify at will
# Last changed: 2013-04-15

__all__ = ["CardiacModel"]

from dolfin import Parameters
from cellmodels import *

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
    solvers (:py:mod:`beatadjoint.splittingsolver`), see their
    documentation for more information on how the attributes are
    interpreted in that context.

    *Arguments*
      domain (:py:class:`dolfin.Mesh`)
        the computational domain in space
      M_i (:py:class:`ufl.Expr`)
        the intra-cellular conductivity as an ufl Expression
      M_e (:py:class:`ufl.Expr`)
        the extra-cellular conductivity as an ufl Expression
      cell_model (:py:class:`~beatadjoint.cellmodels.cardiaccellmodel.CardiacCellModel`)
        a cell model
      applied_current (:py:class:`ufl.Expr`, optional)
        an applied current as an ufl Expression
      stimulus (:py:class:`ufl.Expr`, optional)
        a stimulus as an ufl Expression


    """
    def __init__(self, domain, M_i, M_e, cell_model, applied_current=None,
                 stimulus=None):
        "Create CardiacModel from given input."

        # Store attributes
        self._domain = domain
        self._intracellular_conductivity = M_i
        self._extracellular_conductivity = M_e
        self._cell_model = cell_model
        self._applied_current = applied_current
        self._stimulus = stimulus

    @property
    def applied_current(self):
        "An applied current (:py:class:`ufl.Expr`)"
        return self._applied_current

    @property
    def stimulus(self):
        "A stimulus (:py:class:`ufl.Expr`)"
        return self._stimulus

    def conductivities(self):
        """Return the intracellular and extracellular conductivities
        as a tuple of UFL Expressions.

        *Returns*
        (M_i, M_e) (:py:class:`tuple` of :py:class:`ufl.Expr`)
        """
        return (self.intracellular_conductivity,
                self.extracellular_conductivity)

    @property
    def intracellular_conductivity(self):
        "The intracellular conductivity (:py:class:`ufl.Expr`)."
        return self._intracellular_conductivity

    @property
    def extracellular_conductivity(self):
        "The intracellular conductivity (:py:class:`ufl.Expr`)."
        return self._intracellular_conductivity

    @property
    def domain(self):
        "The spatial domain (:py:class:`dolfin.Mesh`)."
        return self._domain

    @property
    def cell_model(self):
        "The cell model (:py:class:`~beatadjoint.cellmodels.cardiaccellmodel.CardiacCellModel`)."
        return self._cell_model


