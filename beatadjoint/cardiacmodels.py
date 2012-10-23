"""This module contains a base class for cardiac models."""

# Copyright (C) 2012 Marie E. Rognes (meg@simula.no)
# Use and modify at will
# Last changed: 2012-10-23

__all__ = ["CardiacModel"]

from dolfin import Parameters
from cellmodels import *

# ------------------------------------------------------------------------------
# Cardiac models
# ------------------------------------------------------------------------------
class CardiacModel:
    """Base class for cardiac models.

    A stimulation protocol (or combinations) can be specified in two
    ways, by setting 'applied_current' or 'stimulus', respectively.

    Example:

      (a) model.applied_current = Expression("I_a(x, t)", t = t0)
      (b) model.stimulus = Expression("I_s(x, t)", t = t0)

    Example (a) corresponds to a right-hand side in the elliptic
    bidomain equation. Example (b) corresponds to a right-hand side in
    the parabolic bidomain equation.

    In other words, with L_i and L_e denoting the appropriate weighted
    Laplacians, the bidomain + state equations are interpreted as:

    v_t - L_i(v, u) = - I_ion(v, s) + I_s

          L_e(v, u) = I_a

          s_t = F(v, s)
    """
    def __init__(self, cell_model=None, parameters=None):
        "Create cardiac model from given cell model and parameters (optional)."
        self._parameters = self.default_parameters()
        if parameters is not None:
            self._parameters.update(parameters)

        # Use dummy model if no cell_model is given
        if cell_model is None:
            self._cell_model = NoCellModel()
        else:
            self._cell_model = cell_model

        # If applied
        self.applied_current = None
        self.stimulus = None

    def domain(self):
        "Return the spatial domain"
        error("Please overload in subclass")

    def cell_model(self):
        "Return the cell model"
        return self._cell_model

    def parameters(self):
        "Return parameters"
        return self._parameters

    def default_parameters(self):
        "Return default parameters"
        parameters = Parameters("CardiacModel")
        return parameters

