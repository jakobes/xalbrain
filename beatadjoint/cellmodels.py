"""This module contains base classes and standard classes for cardiac
cell models."""

# Copyright (C) 2012 Marie E. Rognes (meg@simula.no)
# Use and modify at will
# Last changed: 2012-10-17

__all__ = ["CardiacCellModel", "FitzHughNagumo", "NoCellModel"]

from dolfin import Parameters, error, Constant

# ------------------------------------------------------------------------------
# Cardiac cell models
# ------------------------------------------------------------------------------
class CardiacCellModel:
    """
    Base class for cardiac cell models. Specialized cell models should
    subclass this class.

    Essentially, a cell model represents a system of ordinary
    differential equations. A cell model is here described by two
    (Python) functions, named F and I. The model describes the
    behaviour of the transmembrane potential 'v' and a number of state
    variables 's'

    The function F gives the right-hand side for the evolution of the
    state variables:

      d/dt s = F(v, s)

    The function I gives the ionic current. If a single cell is
    considered, I gives the (negative) right-hand side for the
    evolution of the transmembrane potential

      d/dt v = - I(v, s)

    If used in a bidomain setting, the ionic current I enters into the
    parabolic partial differential equation of the bidomain equations.
    """

    def __init__(self, parameters=None):
        "Create cardiac cell model, optionally from given parameters"
        self._parameters = self.default_parameters()
        if parameters is not None:
            self._parameters.update(parameters)
        self.applied_current = None

    def default_parameters(self):
        "Set-up and return default parameters"
        parameters = Parameters("CardiacCellModel")
        return parameters

    def initial_conditions(self):
        """
        Return initial conditions of v and s as an Expresson

        Need to to be over loaded to be usefull
        """
        return 

    def before_run(self):
        "Initialize dolfin coefficients from parameters"
        self._coefficient_parameters = {}
        for param, value in self._parameters.items():
            self._coefficient_parameters[param] = Constant(value)

    def parameters(self):
        "Return the current parameters"
        return self._parameters

    def F(self, v, s):
        "Return right-hand side for state variable evolution."
        error("Must define F = F(v, s)")

    def I(self, v, s):
        "Return ionic current."
        error("Must define I = I(v, s)")

    def num_states(self):
        """Return number of state variables (in addition to the
        membrane potential)."""
        error("Must overload num_states")

    def __str__(self):
        "Return string representation of class"
        return "Some cardiac cell model"

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
        # Extract coefficient parameters
        c_1 = self._coefficient_parameters["c_1"]
        c_2 = self._coefficient_parameters["c_2"]
        v_rest = self._coefficient_parameters["v_rest"]
        v_peak = self._coefficient_parameters["v_peak"]
        v_amp = v_peak - v_rest
        v_th = v_rest + self._coefficient_parameters["a"]*v_amp

        # Define current
        i = (c_1/(v_amp**2)*(v - v_rest)*(v - v_th)*(v_peak - v)
             - c_2/(v_amp)*(v - v_rest)*s)
        return - i

    def F(self, v, s):
        # Extract coefficient parameters
        b = self._coefficient_parameters["b"]
        v_rest = self._coefficient_parameters["v_rest"]
        c_3 = self._coefficient_parameters["c_3"]

        # Define model
        return b*(v - v_rest - c_3*s)

    def num_states(self):
        return 1

    def __str__(self):
        return "FitzHugh-Nagumo cardiac cell model"

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

