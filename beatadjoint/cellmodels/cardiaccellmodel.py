"""This module contains a base class for cardiac cell models."""

__author__ = "Marie E. Rognes (meg@simula.no), 2012--2013"

__all__ = ["CardiacCellModel"]

from dolfin import Parameters

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

    (*)  d/dt v = - I(v, s)

    If used in a bidomain setting, the ionic current I enters into the
    parabolic partial differential equation of the bidomain equations.

    If a stimulus is provided via

      cell = CardiacCellModel()
      cell.stimulus = Expression("I_s(t)")

    then I_s is added to the right-hand side of (*), which thus reads

       d/dt v = - I(v, s) + I_s

    Note that the cardiac cell model stimulus is ignored when the cell
    model is used a spatially-varying setting (for instance in the
    bidomain setting). In this case, the user is expected to specify a
    stimulus for the cardiac model instead.
    """

    def __init__(self, params=None):
        "Create cardiac cell model, optionally from given parameters."
        self._parameters = self.default_parameters()
        if params is not None:
            self._parameters.update(params)
        self.stimulus = None

    @staticmethod
    def default_parameters():
        "Set-up and return default parameters."
        params = Parameters("CardiacCellModel")
        return params

    def initial_conditions(self):
        "Return initial conditions for v and s as an Expression."
        error("Must define initial_conditions.")
        return

    def parameters(self):
        "Return the current parameters."
        return self._parameters

    def F(self, v, s, time=None):
        "Return right-hand side for state variable evolution."
        error("Must define F = F(v, s)")

    def I(self, v, s, time=None):
        "Return the ionic current."
        error("Must define I = I(v, s)")

    def num_states(self):
        """Return number of state variables (in addition to the
        membrane potential)."""
        error("Must overload num_states")

    def __str__(self):
        "Return string representation of class."
        return "Some cardiac cell model"
