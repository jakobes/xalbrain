
"""This module contains an equation for testingpurposes."""

from xalbrain.cellmodels import CardiacCellModel
from collections import OrderedDict


class TestCellModel(CardiacCellModel):
    """Class represents the logisitc eqiuation for testing purposes."""

    def __init__(self, params=None,init_conditions=None):
        CardiacCellModel.__init__(self, params,init_conditions)

    def I(self, V, s, time=None):
        return -(2*V + s)

    def F(self, v, s,time=None):
        return v + s

    def num_states(self):
        return 1

    @staticmethod
    def default_initial_conditions():
        """Setup default initial condition."""
        return OrderedDict([("V", 1.0), ("s", 2.0)])
