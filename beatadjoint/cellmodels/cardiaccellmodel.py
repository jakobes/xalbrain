"""This module contains a base class for cardiac cell models."""
from __future__ import division

__author__ = "Marie E. Rognes (meg@simula.no), 2012--2013"
__all__ = ["CardiacCellModel"]

from beatadjoint.dolfinimport import Parameters, Expression, error, GenericFunction
from collections import OrderedDict

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

    def __init__(self, params=None, init_conditions=None):
        """
        Create cardiac cell model

        *Arguments*
         params (dict, :py:class:`dolfin.Mesh`, optional)
           optional model parameters
         init_conditions (dict, :py:class:`dolfin.Mesh`, optional)
           optional initial conditions
        """

        # FIXME: MER: Does this need to be this complicated?
        self._parameters = self.default_parameters()
        self._initial_conditions = self.default_initial_conditions()

        params = params or OrderedDict()
        init_conditions = init_conditions or OrderedDict()

        if params:
            assert isinstance(params, dict), \
                   "expected a dict or a Parameters, as the params argument"
            if isinstance(params, Parameters):
                params = params.to_dict()
            self.set_parameters(**params)

        if init_conditions:
            assert isinstance(init_conditions, dict), \
                "expected a dict or a Parameters, as the init_condition argument"
            if isinstance(init_conditions, Parameters):
                init_conditions = init_conditions.to_dict()
            self.set_initial_conditions(**init_conditions)

        self.stimulus = None

    @staticmethod
    def default_parameters():
        "Set-up and return default parameters."
        return OrderedDict()

    @staticmethod
    def default_initial_conditions():
        "Set-up and return default initial conditions."
        return OrderedDict()

    def set_parameters(self, **params):
        "Update parameters in model"
        for param_name, param_value in params.items():
            if param_name not in self._parameters:
                error("'%s' is not a parameter in %s" %(param_name, self))
            if not isinstance(param_value, (float, int, GenericFunction)):
                error("'%s' is not a scalar or a GenericFunction" % param_name)
            if isinstance(param_value, GenericFunction) and \
               param_value.value_size() != 1:
                error("expected the value_size of '%s' to be 1" % param_name)

            self._parameters[param_name] = param_value

    def set_initial_conditions(self, **init):
        "Update initial_conditions in model"
        for init_name, init_value in init.items():
            if init_name not in self._initial_conditions:
                error("'%s' is not a parameter in %s" %(init_name, self))
            if not isinstance(init_value, (float, int, GenericFunction)):
                error("'%s' is not a scalar or a GenericFunction" % init_name)
            if isinstance(init_value, GenericFunction) and \
               init_value.value_size() != 1:
                error("expected the value_size of '%s' to be 1" % init_name)
            self._initial_conditions[init_name] = init_value

    def initial_conditions(self):
        "Return initial conditions for v and s as an Expression."
        return Expression(self._initial_conditions.keys(), \
                          **self._initial_conditions)

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
