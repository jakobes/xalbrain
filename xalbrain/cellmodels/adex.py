from dolfin import Parameters, Expression
from xalbrain.cellmodels import CellModel

from dolfin import exp

from collections import OrderedDict


__author__ = "Jakob E. Schrein (jakob@xal.no), 2017"


class Adex(CellModel):
    def __init__(self, params=None, init_conditions=None):
        "Create neuronal cell model, optionally from given parameters."
        CellModel.__init__(self, params, init_conditions)

    def I(self, V, w, time=None):
        """Return the ionic current."""
        # Extract parameters
        C = self._parameters["C"]
        g_L = self._parameters["g_L"]
        E_L = self._parameters["E_L"]
        V_T = self._parameters["V_T"]
        Delta_T = self._parameters["Delta_T"]
        b = self._parameters["b"]
        spike = self._parameters["V_T"]

        # FIXME: Add stimulus?
        I = (g_L*Delta_T*exp((V - V_T)/Delta_T) - g_L*(V - E_L) - w)/C
        return -I   # FIXME: Why -1?

    def F(self, V, w, time=None):
        """Return right-hand side for state variable evolution."""

        # Extract parameters
        a = self._parameters["a"]
        E_L = self._parameters["E_L"]
        tau_w = self._parameters["tau_w"]

        # Define model
        F = (a*(V - E_L) - w)/tau_w
        return -F   # FIXME: Why -1?

    @staticmethod
    def default_parameters():
        """Set-up and return default parameters."""
        # TODO: Why am I using OrderedDict?
        params = OrderedDict([
            ("C", 59.0),            # Membrane capacitance (pF)
            ("g_L", 2.9),         # Leak conductance (nS)
            ("E_L", -62.0),       # Leak reversal potential (mV)
            ("V_T", -42.0),       # Spike threshold (mV)
            ("Delta_T", 3.0),     # Slope factor (mV)
            ("a", 16.0),          # Subthreshold adaptation (nS)
            ("tau_w", 144.0),       # Adaptation time constant (ms)
            ("b", 0.061),         # Spike-triggered adaptation (nA)
            ("spike", 20.0)         # When to reset (mV)
        ])
        return params

    #@staticmethod
    def default_initial_conditions(self):
        """Return default intial conditions. FIXME: I have no idea about values."""
        # TODO: Why am I using OrderedDict?
        ic = OrderedDict([
            ("V", self._parameters["E_L"]),
            ("w", 0.0)
        ])
        return ic

    def num_states(self):
        "Return number of state variables."
        return 1

    def update(self, vs):
        """Placeholder for compatibility with the slow version."""
        pass

    def __str__(self):
        "Return string representation of class."
        return "Adex (Manual) -- NB! requires AdexPointIntegralSolver"
