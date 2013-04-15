"""This module contains an abstract base class for cardiac models:
:py:class:`~beatadjoint.cardiacmodels.CardiacModel`.  This class
should be subclassed for setting up specific cardiac simulation
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
    """An abstract base class for cardiac models.

    Subclasses of CardiacModel represent a specific cardiac simulation
    set-up and must provide

      * A cardiac cell model (if any)
      * A computational domain
      * Extra-cellular and intra-cellular conductivities
      * Stimulus

    *Arguments*
      cell_model (:py:class:`~beatadjoint.cellmodels.cardiaccellmodel.CardiacCellModel`)
        a cell model
      parameters (:py:class:`dolfin.Parameters`)
        (optional) a Parameters object controlling solver parameters

    """
    def __init__(self, cell_model=None, parameters=None):
        "Create cardiac model from given cell model and parameters."
        self._parameters = self.default_parameters()
        if parameters is not None:
            self._parameters.update(parameters)

        # Use dummy model if no cell_model is given
        if cell_model is None:
            self._cell_model = NoCellModel()
        else:
            self._cell_model = cell_model

        # If applied
        #self._applied_current = None
        #self._stimulus = None
        self.applied_current = None
        self.stimulus = None

    #@property
    #def applied_current(self):
    #    "Any applied current as a :py:class:`dolfin.GenericFunction`."
    #    return self._applied_current

    #@property
    #def stimulus(self):
    #    "Any stimulus as a :py:class:`dolfin.GenericFunction`."
    #    return self._stimulus

    def conductivities(self):
        """Return the intracellular and extracellular conductivities
        as UFL Expressions

        *Returns*
           (M_i, M_e) (:py:class:`tuple` of :py:class:`ufl.Expr`)
        """

        error("Please overload in subclass.")

    def domain(self):
        """Return the spatial domain as a dolfin Mesh

        *Returns*
           mesh (:py:class:`dolfin.Mesh`)
        """
        error("Please overload in subclass")

    def cell_model(self):
        """Return the cell model as a CardiacCellModel

        *Returns*
           cell model (:py:class:`~beatadjoint.cellmodels.cardiaccellmodel.CardiacCellModel`)
        """
        return self._cell_model

    def parameters(self):
        """Return the current parameters

        *Returns*
           parameters (:py:class:`dolfin.Parameters`)
        """
        return self._parameters

    def default_parameters(self):
        """Return the default parameters

        *Returns*
           default parameters (:py:class:`dolfin.Parameters`)
        """
        parameters = Parameters("CardiacModel")
        return parameters

