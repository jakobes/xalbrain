"""This module contains a dummy cardiac cell models."""

# Copyright (C) 2012 Marie E. Rognes (meg@simula.no)
# Use and modify at will
# Last changed: 2012-10-22

__all__ = ["NoCellModel"]

from cardiaccellmodel import *

# ------------------------------------------------------------------------------
# Dummy cell model
# ------------------------------------------------------------------------------
class NoCellModel(CardiacCellModel):
    """
    Class representing no cell model (only bidomain equations). It
    actually just represents a single completely decoupled ODE.
    """
    def __init__(self, parameters=None):
        CardiacCellModel.__init__(self, parameters)

    def I(self, v, s):
        return 0

    def F(self, v, s):
        # Define model
        return -s

    def num_states(self):
        return 1

    def __str__(self):
        return "No cardiac cell model"

