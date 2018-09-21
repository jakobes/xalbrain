import ufl

import dolfin as df

from xalbrain.cellmodels import CardiacCellModel

from collections import OrderedDict

from typing import (
    Dict,
)


class Cressman(CardiacCellModel):

    def __init__(self, params: df.parameters=None, init_conditions: Dict[str, float]=None) -> None:
        """Create neuronal cell model, optionally from given parameters.

        See Cressman TODO: Look this up
        """
        CardiacCellModel.__init__(self, params, init_conditions)

    @staticmethod
    def default_parameters():
        """Default parameters from Andre Erhardt. 

        TODO: add units and explanation
        """
        params = OrderedDict([
            ("Cm", 1),
            ("GNa", 100),
            ("GK", 40),
            ("GAHP", 0.01),
            ("GKL", 0.05),
            ("GNaL", 0.0175),
            ("GClL", 0.05),
            ("GCa", 0.1),
            ("Gglia", 66),
            ("Koinf", 4),     # Default = 4
            ("gamma1", 0.0445),
            ("tau", 1000),
            ("control", 1),
            ("period", 1000),
            ("duration", 300),
            ("amplitude", 3)
        ])
        return params

    def default_initial_conditions(self):
        ic = OrderedDict([
            ("V", -50),
            ("m", 0.0936),
            ("h", 0.96859),
            ("n", 0.08553),
            ("foo", 0.0),
            ("bar", 7.8),
            ("baz", 15.5),
        ])
        return ic

    def I(self, V, s, time=None):
        """dv/dt = -I."""
        Cm = self._parameters["Cm"]
        GNa = self._parameters["GNa"]
        GNaL = self._parameters["GNaL"]
        GK = self._parameters["GK"]
        GKL = self._parameters["GKL"]
        GAHP = self._parameters["GAHP"]
        control = self._parameters["control"]
        GClL = self._parameters["GClL"]

        # Define some parameters
        beta0 = 7
        Cli = 6
        Clo = 130

        Nao = 144 - beta0*(s[5] - 18)
        ENa = 26.64*df.ln(Nao/s[5])
        Ki = 140 + (18 - s[5])
        EK = 26.64*df.ln(control*s[4]/Ki)
        ECl = 26.64*df.ln(Cli/Clo)

        INa = GNa*s[0]**3*s[2]*(V - ENa) + GNaL*(V - ENa)
        IK = (GK*s[1]**4 + GAHP*s[3]/(1 + s[3]) + GKL)*(V - EK)
        ICl = GClL*(V - ECl)
        return (INa + IK + ICl)/Cm       # Check the sign
        # return -(INa + IK + ICl)/Cm       # Check the sign

    def F(self, V, s, time=None):
        """ds/dt = F(v, s)."""
        GCa = self._parameters["GCa"]
        gamma1 = self._parameters["gamma1"]
        Gglia = self._parameters["Gglia"]
        tau = self._parameters["tau"]
        control = self._parameters["control"]
        GNa = self._parameters["GNa"]
        GNaL = self._parameters["GNaL"]
        GK = self._parameters["GK"]
        GKL = self._parameters["GKL"]
        GAHP = self._parameters["GAHP"]
        Koinf = self._parameters["Koinf"]

        # Define some parameters
        ECa = 120
        phi = 3
        rho = 1.25
        eps0 = 1.2
        beta0 = 7
        Cli = 6
        Clo = 130

        Nao = 144 - beta0*(s[5] - 18)
        ENa = 26.64*df.ln(Nao/s[5])
        Ki = 140 + (18 - s[5])
        EK = 26.64*df.ln(control*s[4]/Ki)
        INa = GNa*s[0]**3*s[2]*(V - ENa) + GNaL*(V - ENa)
        IK = (GK*s[1]**4 + GAHP*s[3]/(1 + s[3]) + GKL)*(V - EK)

        a_m = (3.0 + (0.1)*V)*(1 - df.exp(-3 -1/10*V))**(-1)
        b_m = 4*df.exp(-55/18 - 1/18*V)
        ah = (0.07)*df.exp(-11/5 - 1/20*V)
        bh = (1 + df.exp(-7/5 - 1/10*V))**(-1)
        an = (0.34 + (0.01)*V)*(1 - df.exp(-17/5 - 1/10*V))**(-1)
        bn = (0.125)*df.exp(-11/20 - 1/80*V)

        taum = (a_m + b_m)**(-1)
        minf = a_m*(a_m + b_m)**(-1)
        h_inf = (bh + ah)**(-1)*ah
        tauh = (bh + ah)**(-1)
        ninf = (an + bn)**(-1)*an
        taun = (an + bn)**(-1)

        dot_m = phi*(minf - s[0])*taum**(-1)
        dot_h = phi*(ninf - s[1])/taun
        dot_n = phi*(h_inf - s[2])/tauh

        Ipump = rho*(1/(1 + df.exp((25 - s[5])/3)))*(1/(1 + df.exp(5.5 - s[4])))
        IGlia = Gglia/(1 + df.exp((18 - s[4])/2.5))
        Idiff = eps0*(s[4] - Koinf)

        dot_foo = -s[3]/80 - 0.002*GCa*(V - ECa)/(1 + df.exp(-(V + 25)/2.5))
        dot_bar = (gamma1*beta0*IK - 2*beta0*Ipump - IGlia - Idiff)/tau
        dot_baz = (gamma1*INa + 3*Ipump)/tau

        F_expressions = [ufl.zero()]*self.num_states()
        F_expressions[0] = dot_m
        F_expressions[1] = dot_h
        F_expressions[2] = dot_n
        F_expressions[3] = dot_foo
        F_expressions[4] = dot_bar
        F_expressions[5] = dot_baz
        return df.as_vector(F_expressions)

    def num_states(self):
        """Return number of state variables."""
        return 6
