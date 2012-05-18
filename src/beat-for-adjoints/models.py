from dolfin import *

# ------------------------------------------------------------------------------
# Cardiac models
# ------------------------------------------------------------------------------

class CardiacModel:
    def __init__(self, cell_model, parameters=None):
        self._parameters = self.default_parameters()
        if parameters is not None:
            self._parameters.update(parameters)
        self._cell_model = cell_model

    def domain(self):
        error("Please overload in subclass")

    def cell_model(self):
        return self._cell_model

    def default_parameters(self):
        parameters = Parameters("CardiacModel")
        return parameters

# ------------------------------------------------------------------------------
# Cardiac cell models
# ------------------------------------------------------------------------------

class CardiacCellModel:

    def __init__(self, parameters=None):
        self._parameters = self.default_parameters()
        if parameters is not None:
            self._parameters.update(parameters)

    def default_parameters(self):
        parameters = Parameters("CardiacCellModel")
        return parameters

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

    def __init__(self, parameters):
        CardiacCellModel.__init__(self, parameters)
        self._epsilon = self._parameters["epsilon"]
        self._gamma = self._parameters["gamma"]
        self._alpha = self._parameters["alpha"]

    def default_parameters(self):
        parameters = Parameters("FitzHughNagumo")
        parameters.add("epsilon", 0.01)
        parameters.add("gamma", 0.5)
        parameters.add("alpha", 0.1)
        return parameters

    def F(self, v, s):
        return self._epsilon*(v - self._gamma*s)

    def I(self, v, s):
        return v*(v - self._alpha)*(1 - v) - s

    def num_states(self):
        return 1

    def __str__(self):
        return "FitzHugh-Nagumo cardiac cell model"
