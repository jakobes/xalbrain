from cbcbeat.dolfinimport import Parameters, Expression
from cbcbeat.cellmodels import CardiacCellModel

from dolfin import exp, assign, conditional, lt
from collections import OrderedDict

import numpy as np


class Test_adex(CardiacCellModel):
    def __init__(self, params=None, init_conditions=None):
        "Create neuronal cell model, optionally from given parameters."
        CardiacCellModel.__init__(self, params, init_conditions)

    def I(self, V, w, time=None):
        # return -np.exp(-float(time))
        return exp(-time)

    def F(self, V, w, time=None):
        return -0.1

    @staticmethod
    def default_parameters():
        "Set-up and return default parameters."
        params = OrderedDict([("C", 281),           # Membrane capacitance (pF)
                              ("g_L", 30),          # Leak conductance (nS)
                              ("E_L", -70.6),       # Leak reversal potential (mV)
                              ("V_T", -50.4),       # Spike threshold (mV)
                              ("Delta_T", 2),       # Slope factor (mV)
                              ("tau_w", 144),       # Adaptation time constant (ms)
                              ("a", 4),             # Subthreshold adaptation (nS)
                              ("spike", 20),        # When to reset (mV)
                              ("b", 0.0805)])       # Spike-triggered adaptation (nA)
        return params

    #@staticmethod
    def default_initial_conditions(self):
        """ Return default intial conditions. FIXME: I have no idea about values
        """
        ic = OrderedDict([("V", 1.0),
                          ("w", 1.0)])
        return ic

    def num_states(self):
        "Return number of state variables."
        return 1

    def __str__(self):
        "Return string representation of class."
        return "(Manual) AdEx neuronal cell model"
