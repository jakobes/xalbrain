"""This module contains a container class for cardiac models:
:py:class:`~beatadjoint.cardiacmodels.CardiacModel`.  This class
should be instantiated for setting up specific cardiac simulation
scenarios.
"""

# Copyright (C) 2012 Marie E. Rognes (meg@simula.no)
# Use and modify at will
# Last changed: 2014-09-12

__all__ = ["CardiacModel"]


from dolfinimport import Parameters, Mesh, Constant
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
      time (:py:class:`dolfin.Constant` or None )
        A constant holding the current time.
      M_i (:py:class:`ufl.Expr`)
        the intra-cellular conductivity as an ufl Expression
      M_e (:py:class:`ufl.Expr`)
        the extra-cellular conductivity as an ufl Expression
      cell_models (:py:class:`~beatadjoint.cellmodels.cardiaccellmodel.CardiacCellModel`)
        a cell model or a dict with cell models associated with a cell model domain
      stimulus (:py:class:`dict`, optional)
        A typically time-dependent external stimulus given as a dict,
        with domain markers as the key and a
        :py:class:`dolfin.Expression` as values. NB: it is assumed
        that the time dependence of I_s is encoded via the 'time'
        Constant.
      applied_current (:py:class:`ufl.Expr`, optional)
        an applied current as an ufl Expression
      cell_model_domains (:py:class:`dolfin.MeshFunction`, optional)
        a mesh function mapping what domain a certain cell model is associated with

    """
    def __init__(self, domain, time, M_i, M_e, cell_models, stimulus=None,
                 applied_current=None, cell_model_domains=None):
        "Create CardiacModel from given input."

        # Check some input
        assert isinstance(domain, Mesh), \
            "Expecting domain to be a Mesh instance, not %r" % domain
        assert isinstance(time, Constant) or time is None, \
            "Expecting time to be a Constant instance (or None)."

        if isinstance(cell_models, dict):
            if not all(isinstance(key, int) for key in cell_models):
                raise TypeError("Expected keys on cell_models to be integers")
            if len(cell_models) == 1:
                cell_models = {None:cell_models.values()[0]}
        else:
            cell_models = {None:cell_models}

        # if no cell_model domains is passed we also expect only one cell_model
        if cell_model_domains is None and (len(cell_models) > 1 or \
                                           cell_models.keys()[0] is not None):
            raise ValueError("When no cell_model_domains are given we only "\
                             "expect one cell model")

        # Store attributes
        self._domain = domain
        self._time = time
        self._intracellular_conductivity = M_i
        self._extracellular_conductivity = M_e
        self._cell_models = cell_models
        self._stimulus = stimulus
        self._applied_current = applied_current
        self._cell_model_domains = cell_model_domains

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
        return self._extracellular_conductivity

    @property
    def time(self):
        "The current time (:py:class:`dolfin.Constant` or None)."
        return self._time

    @property
    def domain(self):
        "The spatial domain (:py:class:`dolfin.Mesh`)."
        return self._domain

    def cell_model(self, domain=None):
        "The cell model (:py:class:`~beatadjoint.cellmodels.cardiaccellmodel.CardiacCellModel`)."
        return self._cell_models[domain]

    @property
    def cell_models(self):
        "The cell model (:py:class:`dict`)."
        return self._cell_models

    @property
    def cell_model_domains(self):
        "The cell model domains (:py:class:``dolfin.MeshFunction``)."
        return self._cell_model_domains
