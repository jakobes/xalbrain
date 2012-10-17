"""This module contains a base class for cardiac models."""

# Copyright (C) 2012 Marie E. Rognes (meg@simula.no)
# Use and modify at will
# Last changed: 2012-10-09

__all__ = ["CardiacModel"]

from dolfin import Parameters
from cellmodels import *

# ------------------------------------------------------------------------------
# Cardiac models
# ------------------------------------------------------------------------------
class CardiacModel:
    "Base class for cardiac models."
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

        self.applied_current = None

    def before_run(self):
        "Initialize model before a run"

        # Init cell model
        self._cell_model.before_run()

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

