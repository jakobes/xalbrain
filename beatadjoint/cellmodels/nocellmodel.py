"""This module contains a dummy cardiac cell model."""

__author__ = "Marie E. Rognes (meg@simula.no), 2012--2013"
__all__ = ["NoCellModel"]

from beatadjoint.cellmodels import CardiacCellModel

# FIXME: This class represents a design flaw rather than anything
# else. Remove in a clean-up of the solvers.

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
        return -s

    def num_states(self):
        return 1

    def __str__(self):
        return "No cardiac cell model"

