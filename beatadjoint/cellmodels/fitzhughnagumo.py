
import ufl

from dolfin import *
from dolfin_adjoint import *

from beatadjoint.cellmodels import CardiacCellModel

class Fitzhughnagumo(CardiacCellModel):
    """
NOT_IMPLEMENTED
    """
    def __init__(self, params=None):
        CardiacCellModel.__init__(self, params)

    def default_parameters(self):
        params = Parameters("Fitzhughnagumo")
        params.add("a", 0.13)
        params.add("b", 0.013)
        params.add("c_1", 0.26)
        params.add("c_2", 0.1)
        params.add("c_3", 1.0)
        params.add("v_peak", 40.0)
        params.add("v_rest", -85.0)
        return params

    def I(self, v, s):
        """
        Transmembrane current
        """
        # Imports
        # No imports for now

        # Assign states
        states = s
        assert(len(states) == 1)
        s, = states

        # Assign parameters
        a = self._parameters["a"]
        c_1 = self._parameters["c_1"]
        c_2 = self._parameters["c_2"]
        v_rest = self._parameters["v_rest"]
        v_peak = self._parameters["v_peak"]

        current = -c_1*(-v + v_peak)*(v - v_rest)*(-a*(v_peak - v_rest) + v -\
            v_rest)/(v_peak - v_rest*v_peak - v_rest) + c_2*s*(v -\
            v_rest)/(v_peak - v_rest)


        return current

    def F(self, v, s):
        """
        Right hand side for ODE system
        """
        # Imports
        # No imports for now

        # Assign states
        states = s
        assert(len(states) == 1)
        s, = states

        # Assign parameters
        c_3 = self._parameters["c_3"]
        b = self._parameters["b"]
        v_rest = self._parameters["v_rest"]

        F_expressions = [\

            b*(-c_3*s + v - v_rest),
            ]

        return as_vector(F_expressions)

    def initial_conditions(self):
        ic = Expression(["V", "S"],\
            V=-85.0, S = 0.0)

        return ic

    def num_states(self):
        return 1

    def __str__(self):
        return 'Fitzhughnagumo cardiac cell model'
