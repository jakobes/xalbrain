import ufl

import dolfin as df

from xalbrain.cellmodels import CardiacCellModel

from collections import OrderedDict

from typing import (
    Dict,
)


class Noble(CardiacCellModel):

    def __init__(self, params: df.parameters=None, init_conditions: Dict[str, float]=None) -> None:
        """Create neuronal cell model, optionally from given parameters.

        See Cressman TODO: Look this up
        """
        CardiacCellModel.__init__(self, params, init_conditions)

    def default_initial_conditions(self) -> OrderedDict:
        ic = OrderedDict([
            ("V", -40.8454),
            ("m", 0.0268),
            ("n", 0.3233),
            ("h", 0.5852)
        ])
        return ic

    @staticmethod
    def default_parameters():
        ic = OrderedDict([
            ("Cm", 12.0),
            ("gNa", 400),
            ("gK1", 1.2),
            ("gL", 0.1845),
            ("gK2", 1.2)
        ])
        return ic

    def I(self, V, s, time=None):
        """dv/dt = -I."""
        Cm = self._parameters["Cm"]
        gNa = self._parameters["gNa"]
        gK1 = self._parameters["gK1"]
        gK2 = self._parameters["gK2"]
        gL = self._parameters["gL"]

        INa = (gNa*s[1]**3*s[0] + 0.14)*(V - 40)
        IK = (gK1*s[2]**4 + gK2*df.exp(-(V + 90)/50) + gK2/80*df.exp((V + 90)/60))*(V + 100)
        IL = gL*(V + 60)
        return (INa + IK + IL)/Cm       # Check sign

    def F(self, V, s, time=None):
        """ds/dt = F(v, s)."""
        am = 0.1*(V + 48)/(1 - df.exp(-(V + 48)/15))
        bm = 0.12*(V + 8)/(df.exp((V + 8)/5)-1)

        an = 0.0001*(V + 50)/(1 - df.exp(-(V + 50)/10))
        bn = 0.002*df.exp(-(V + 90)/80)

        ah = 0.17*df.exp(-(V + 90)/20)
        bh = 1/(1 + df.exp(-(V + 42)/10))

        tau_m = 1/(am + bm)
        m_inf = am/(am + bm)

        tau_n = 1/(an + bn)
        n_inf = an/(an + bn)

        tau_h = 1/(ah + bh)
        h_inf = ah/(ah + bh)

        F_expressions = [ufl.zero()]*self.num_states()
        F_expressions[0] = (h_inf - s[0])/tau_h
        F_expressions[1] = (m_inf - s[1])/tau_m
        F_expressions[2] = (n_inf - s[2])/tau_n
        return df.as_vector(F_expressions)

    def num_states(self):
        """Return number of state variables."""
        return 3
