"""This module contains a base class for cardiac models."""

# Copyright (C) 2012 Marie E. Rognes (meg@simula.no)
# Use and modify at will
# Last changed: 2012-09-25

__all__ = ["CardiacModel"]

from dolfin import Parameters

# ------------------------------------------------------------------------------
# Cardiac models
# ------------------------------------------------------------------------------
class CardiacModel:
    "Base class for cardiac models."
    def __init__(self, cell_model, parameters=None):
        "Create cardiac model from given cell model and parameters (optional)."
        self._parameters = self.default_parameters()
        if parameters is not None:
            self._parameters.update(parameters)
        self._cell_model = cell_model
        self.applied_current = None

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

