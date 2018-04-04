"""This module contains the logistic equation for testingpurposes."""

from xalbrain.cellmodels import CardiacCellModel
from collections import OrderedDict


class LogisticCellModel(CardiacCellModel):
    """Class represents the logisitc eqiuation for testing purposes."""

    def __init__(self, params=None,init_conditions=None):
        CardiacCellModel.__init__(self, params,init_conditions)

    def I(self, V, s, time=None):
        return -V*(1 - V)

    def F(self, v, s,time=None):
        return -s

    def num_states(self):
        return 1

    @staticmethod
    def default_initial_conditions():
        """Setup default initial condition."""
        return OrderedDict([("V", 0.0001), ("s", 0.0)])
