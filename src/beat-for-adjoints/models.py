from dolfin import *

# ------------------------------------------------------------------------------
# Cardiac models
# ------------------------------------------------------------------------------

class CardiacModel:
    def __init__(self, domain, cell, parameters=None):
        self._parameters = parameters
        self._domain = domain
        self._cell_model = cell

    def mesh(self):
        return self._domain

    def cell_model(self):
        return self._cell_model

# ------------------------------------------------------------------------------
# Cardiac cell models
# ------------------------------------------------------------------------------

class CardiacCellModel:
    def __init__(self, parameters=None):
        self._parameters = parameters

    def F(self, v, s):
        error("Must define F = F(v, s)")

    def I(self, v, s):
        error("Must define I = I(v, s)")

    def num_states(self):
        "Return number of state variables (in addition to membrane potential)."
        error("Must overload num_states")

    def __str__(self):
        return "Some cardiac cell model"

class FitzHughNagumo(CardiacCellModel):

    def __init__(self, parameters):
        CardiacCellModel.__init__(self, parameters)
        self._epsilon = self._parameters["epsilon"]
        self._gamma = self._parameters["gamma"]
        self._alpha = self._parameters["alpha"]

    def F(self, v, s):
        return self._epsilon*(v - self._gamma*s)

    def I(self, v, s):
        return v*(v - self._alpha)*(1 - v) - s

    def num_states(self):
        "Return number of state variables (in addition to membrane potential)."
        return 1

    def __str__(self):
        return "FitzHugh-Nagumo cardiac cell model"
