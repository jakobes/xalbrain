"""This module contains a FitzHugh-Nagumo cardiac cell model

The module was written by hand, in particular it was not
autogenerated.
"""

from __future__ import division

__author__ = "Marie E. Rognes (meg@simula.no), 2012--2013"
__all__ = ["FitzHughNagumoManual"]

from collections import OrderedDict
from dolfin import Parameters, Expression
from xalbrain.cellmodels import CellModel


class FitzHughNagumoManual(CellModel):
    """
    A reparametrized FitzHughNagumo model, based on Section 2.4.1 in
    "Computing the electrical activity in the heart" by Sundnes et al,
    2006.

    This is a model containing two nonlinear, ODEs for the evolution
    of the transmembrane potential v and one additional state variable
    s.
    """
    def __init__(self, params=None, init_conditions=None):
        "Create cardiac cell model, optionally from given parameters."
        CellModel.__init__(self, params, init_conditions)

    @staticmethod
    def default_parameters():
        "Set-up and return default parameters."
        params = OrderedDict([("a", 0.13),
                              ("b", 0.013),
                              ("c_1", 0.26),
                              ("c_2", 0.1),
                              ("c_3", 1.0),
                              ("v_peak", 40.0),
                              ("v_rest", -85.0)])
        return params

    def I(self, v, s, time=None):
        "Return the ionic current."
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

    def F(self, v, s, time=None):
        "Return right-hand side for state variable evolution."

        # Extract parameters
        b = self._parameters["b"]
        v_rest = self._parameters["v_rest"]
        c_3 = self._parameters["c_3"]

        # Define model
        return b*(v - v_rest - c_3*s)

    @staticmethod
    def default_initial_conditions():
        ic = OrderedDict([("V", -85.0),
                          ("S", 0.0)])
        return ic

    def num_states(self):
        "Return number of state variables."
        return 1

    def __str__(self):
        "Return string representation of class."
        return "(Manual) FitzHugh-Nagumo cardiac cell model"

