# Copyright (C) 2012 Marie E. Rognes (meg@simula.no)
# Use and modify at will
# Last changed: 2012-09-24

__all__ = ["CardiacModel", "CardiacCellModel", "FitzHughNagumo",
           "NoCellModel"]

from dolfin import Parameters

# ------------------------------------------------------------------------------
# Cardiac models
# ------------------------------------------------------------------------------

class CardiacModel:
    def __init__(self, cell_model, parameters=None):
        self._parameters = self.default_parameters()
        if parameters is not None:
            self._parameters.update(parameters)
        self._cell_model = cell_model
        self.applied_current = None

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

# ------------------------------------------------------------------------------
# Cardiac cell models
# ------------------------------------------------------------------------------

class CardiacCellModel:
    """
    The cell model is described by two functions F and I: modelling a
    system of ODEs of the form:

    d/dt s = F(v, s)

    where v is the transmembrane potential and s is the state
    variable(s)

    Further, I(v, s) gives the ionic current (in the bi/monodomain
    equations)
    """

    def __init__(self, parameters=None):
        self._parameters = self.default_parameters()
        if parameters is not None:
            self._parameters.update(parameters)
        self.applied_current = None

    def default_parameters(self):
        "Return default parameters"
        parameters = Parameters("CardiacCellModel")
        return parameters

    def parameters(self):
        "Return parameters"
        return self._parameters

    def F(self, v, s):
        error("Must define F = F(v, s)")

    def I(self, v, s):
        error("Must define I = I(v, s)")

    def num_states(self):
        "Return number of state variables (in addition to membrane potential)."
        error("Must overload num_states")

    def __str__(self):
        "Return string representation of class"
        return "Some cardiac cell model"

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

