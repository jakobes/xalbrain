"""This module contains a FitzHugh-Nagumo cardiac cell models."""

# Copyright (C) 2012 Marie E. Rognes (meg@simula.no)
# Use and modify at will
# Last changed: 2012-10-22

__all__ = ["FitzHughNagumo"]

from dolfin import Parameters
from cardiaccellmodel import *

# ------------------------------------------------------------------------------
# FitzHughNagumo model
# ------------------------------------------------------------------------------
class FitzHughNagumo(CardiacCellModel):
    """
    Reparametrized FitzHughNagumo model
    (cf. 2.4.1 in Sundnes et al 2006)
    """
    def __init__(self, parameters=None):
        CardiacCellModel.__init__(self, parameters)

    def default_parameters(self):
        parameters = Parameters("FitzHughNagumo")
        parameters.add("a", 0.13)
        parameters.add("b", 0.013)
        parameters.add("c_1", 0.26)
        parameters.add("c_2", 0.1)
        parameters.add("c_3", 1.0)
        parameters.add("v_rest", -85.)
        parameters.add("v_peak", 40.)
        return parameters

    def I(self, v, s):
        # Extract parameters
        c_1 = self._parameters["c_1"]
        c_2 = self._parameters["c_2"]
        v_rest = self._parameters["v_rest"]
        v_peak = self._parameters["v_peak"]
        v_amp = v_peak - v_rest
        v_th = v_rest + self._parameters["a"]*v_amp

        # Define current
        i = (c_1/(v_amp**2)*(v - v_rest)*(v - v_th)*(v_peak - v)
             - c_2/(v_amp)*(v - v_rest)*s)
        return - i

    def F(self, v, s):
        # Extract parameters
        b = self._parameters["b"]
        v_rest = self._parameters["v_rest"]
        c_3 = self._parameters["c_3"]

        # Define model
        return b*(v - v_rest - c_3*s)

    def num_states(self):
        return 1

    def __str__(self):
        return "FitzHugh-Nagumo cardiac cell model"

