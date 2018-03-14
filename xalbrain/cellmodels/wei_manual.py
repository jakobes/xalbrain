from xalbrain.dolfinimport import Parameters, Expression
from xalbrain.cellmodels import CardiacCellModel

import dolfin
import ufl
from dolfin import ln, exp, pi, ge, conditional

from collections import OrderedDict


class Wei(CardiacCellModel):
    def __init__(self, params=None, init_conditions=None):
        """
        Create neuronal cell model, optionally from given parameters.

        See Wei, Ullah, Schiff, 2014.
        """
        CardiacCellModel.__init__(self, params, init_conditions)

    @staticmethod
    def default_parameters():
        params = OrderedDict([
            ("C", 1),           # [muF/cm^2]        Membrane capacitance
            ("G_Na", 30),       # [mS/cm^2]
            ("G_NaL", 0.0247),  # [mS/cm^2]
            ("G_K", 25),        # [mS/cm^2]
            ("G_KL", 0.05),     # [mS/cm^2]
            ("G_ClL", 0.1),     # [mS/cm^2]
            ("beta0", 7),       # Ratio of initial extra- and intracellular volume
            ("rho_max", 0.8),   # [mM/s]
            ("tau", 1e-3),      # [s???] time constant
            ("Ukcc2", 0.3),     # [mM/s] Maximal KCC2 cotransporter strength
            ("Unkcc1", 0.1),    # [mM/s] Maximal KCC2 cotransporter strength
            ("eps_K", 0.25),    # [1/s] Potassium diffusion coefficient
            ("G_glia", 5),      # [mM/s] Maximal glia uptake strength of potassium
            ("eps_O", 0.17),    # [1/s] Oxygen diffusion rate
            ("KBath", 8.5),     # Extracellular potassium concentration? 
            ("OBath", 32),      # Extracellular oxygen concentration?

            ])
        return params

    def default_initial_conditions(self):
        beta0 = self._parameters["beta0"]
        vol = 1.4368e-15         # TODO: FIXME!!!!!
        volo = 1/beta0*vol
        ic = OrderedDict([
            ("V", -74.30),          # [mV] Membrane potential
            ("m", 0.0031),          # Voltage gating? 
            ("h", 0.9994),          # Voltage gating? 
            ("n", 0.0107),          # Voltage gating? 
            ("Nko", 4*volo),        # [mol?] Initial extracellular potassium number
            ("NKi", 140*vol),       # [mol?] Initial intracellular potassium number
            ("NNao", 144*volo),     # [mol?] Initial extracellular sodium number
            ("NNai", 18*vol),       # [mol?] Initial intracellular sodium number
            ("NClo", 130*volo),     # [mol?] Initial extracellular chloride number
            ("NCli", 6*vol),        # [mol?] Initial intracellular chloride number
            ("vol", vol),           # [m^3] Initial intracellular volume
            ("O", 29.3),            # [mol?] Initial oxygen concentration
            ])
        return ic

    def _get_concentrations(self, V, s):
        m, h, n, NKo, NKi, NNao, NNai, NClo, NCli, voli, O = s
        beta0 = self._parameters["beta0"]
        ic = self.default_initial_conditions()

        volo = (1 + 1/beta0)*ic["vol"] - voli
        Ko = NKo/volo
        Ki = NKi/voli
        Nao = NNao/volo
        Nai = NNai/voli
        Clo = NClo/volo
        Cli = NCli/voli
        return Ko, Ki, Nao, Nai, Clo, Cli

    def _gamma(self, voli):
        alpha = 4*pi*(3*voli/(4*pi))**(2/3)
        F = 96485.33
        return alpha/(F*voli)*1e-2

    def _I_K(self, V, n, Ko, Ki):
        G_K = self._parameters["G_K"]
        G_KL = self._parameters["G_KL"]
        E_K = 26.64*ln(Ko/Ki)
        return G_K*n**4*(V - E_K) + G_KL*(V - E_K)

    def _I_Na(self, V, m, h, Nao, Nai):
        G_Na = self._parameters["G_Na"]
        G_NaL = self._parameters["G_NaL"]
        E_Na = 26.64*ln(Nao/Nai)
        return G_Na*m**3*h*(V - E_Na) + G_NaL*(V - E_Na)

    def _I_Cl(self, V, Clo, Cli):
        G_ClL = self._parameters["G_ClL"]
        E_Cl = 26.64*ln(Cli/Clo)
        return G_ClL*(V - E_Cl)

    def _I_pump(self, Nai, Ko, O, gamma):
        rho_max = self._parameters["rho_max"]
        rho = rho_max/(1 + exp((20 - O)/3))/gamma
        return rho/(1 + exp((25 - Nai)/3))*1/(1 + exp(3.5 - Ko))

    def I(self, V, s, time=None):
        m, h, n, NKo, NKi, NNao, NNai, NClo, NCli, voli, O = s
        O = dolfin.conditional(dolfin.ge(O, 0), O, 0)
        vol = 1.4368e-15    # unit:m^3, when r=7 um,v=1.4368e-15 m^3
        beta0 = 7
        volo = (1 + 1/beta0)*vol - voli     # Extracellular volume
        C = 1

        Ko, Ki, Nao, Nai, Clo, Cli = self._get_concentrations(V, s)

        I_K = self._I_K(V, n, Ko, Ki)
        I_Na = self._I_Na(V, m, h, Nao, Nai)
        I_Cl = self._I_Cl(V, Clo, Cli)
        gamma = self._gamma(voli)
        I_pump = self._I_pump(Nai, Ko, O, gamma)
        return (I_Na + I_K + I_Cl + I_pump)/C       # Sign is correcct I think

    def F(self, V, s, time=None):
        m, h, n, NKo, NKi, NNao, NNai, NClo, NCli, voli, O = s
        O = dolfin.conditional(dolfin.ge(O, 0), O, 0)

        G_K = self._parameters["G_K"]
        G_Na = self._parameters["G_Na"]
        G_ClL = self._parameters["G_ClL"]
        G_Kl = self._parameters["G_KL"]
        G_NaL = self._parameters["G_NaL"]
        C = self._parameters["C"]

        # Ion Concentration related Parameters
        eps_K = self._parameters["eps_K"]
        G_glia = self._parameters["G_glia"]
        eps_O = self._parameters["eps_O"]
        Ukcc2 = self._parameters["Ukcc2"] 
        Unkcc1 = self._parameters["Unkcc1"]
        rho_max = self._parameters["rho_max"]

        # Volume 
        vol = 1.4368e-15    # unit:m^3, when r=7 um,v=1.4368e-15 m^3    # TODO: add to parameters
        beta0 = self._parameters["beta0"]

        # Time Constant
        tau = self._parameters["tau"]
        KBath = self._parameters["KBath"]
        OBath = self._parameters["OBath"]

        gamma = self._gamma(voli)
        volo = (1 + 1/beta0)*vol - voli     # Extracellular volume
        beta = voli/volo                    # Ratio of intracelluar to extracelluar volume

        # Gating variables
        alpha_m = 0.32*(54 + V)/(1 - exp(-(V + 54)/4))
        beta_m = 0.28*(V + 27)/(exp((V + 27)/5) - 1)

        alpha_h = 0.128*exp(-(V + 50)/18)
        beta_h = 4/(1 + exp(-(V + 27)/5))

        alpha_n = 0.032*(V + 52)/(1 - exp(-(V + 52)/5))
        beta_n = 0.5*exp(-(V + 57)/40)

        dotm = alpha_m*(1 - m) - beta_m*m
        doth = alpha_h*(1 - h) - beta_h*h
        dotn = alpha_n*(1 - n) - beta_n*n

        Ko, Ki, Nao, Nai, Clo, Cli = self._get_concentrations(V, s)

        fo = 1/(1 + exp((2.5 - OBath)/0.2))
        fv = 1/(1 + exp((beta - 20)/2))
        eps_K *= fo*fv
        G_glia *= fo

        rho = rho_max/(1 + exp((20 - O)/3))/gamma
        I_glia = G_glia/(1 + exp((18 - Ko)/2.5))
        Igliapump = rho/3/(1 + exp((25 - 18)/3))/(1 + exp(3.5 - Ko))
        I_diff = eps_K*(Ko - KBath) + I_glia + 2*Igliapump*gamma

        I_K = self._I_K(V, n, Ko, Ki)
        I_Na = self._I_Na(V, m, h, Nao, Nai)
        I_Cl = self._I_Cl(V, Clo, Cli)
        I_pump = self._I_pump(Nai, Ko, O, gamma)

        # Cloride transporter (mM/s)
        fKo = 1/(1 + exp(16 - Ko))
        FKCC2 = Ukcc2*ln((Ki*Cli)/(Ko*Clo))
        FNKCC1 = Unkcc1*fKo*(ln((Ki*Cli)/(Ko*Clo)) + ln((Nai*Cli)/(Nao*Clo)))

        dotNKo = tau*volo*(gamma*beta*(I_K -  2.0*I_pump) - I_diff + FKCC2*beta + FNKCC1*beta)
        dotNKi = -tau*voli*(gamma*(I_K - 2.0*I_pump) + FKCC2 + FNKCC1)

        dotNNao = tau*volo*beta*(gamma*(I_Na + 3.0*I_pump) + FNKCC1)
        dotNNai = -tau*voli*(gamma*(I_Na + 3.0*I_pump) + FNKCC1)

        dotNClo = beta*volo*tau*(FKCC2 - gamma*I_Cl + 2*FNKCC1)
        dotNCli = voli*tau*(gamma*I_Cl - FKCC2 - 2*FNKCC1)

        r1 = vol/voli
        r2 = 1/beta0*vol/((1 + 1/beta0)*vol - voli)
        pii = Nai + Cli + Ki + 132*r1
        pio = Nao + Ko + Clo + 18*r2

        vol_hat = vol*(1.1029 - 0.1029*exp((pio - pii)/20))
        dotVoli = -(voli - vol_hat)/0.25*tau
        dotO = tau*(-5.3*(I_pump + Igliapump)*gamma + eps_O*(OBath - O))

        F_expressions = [ufl.zero()]*self.num_states()
        F_expressions[0] = dotm
        F_expressions[1] = doth
        F_expressions[2] = dotn
        F_expressions[3] = dotNKo
        F_expressions[4] = dotNKi
        F_expressions[5] = dotNNao
        F_expressions[6] = dotNNai
        F_expressions[7] = dotNClo
        F_expressions[8] = dotNCli
        F_expressions[9] = dotVoli
        F_expressions[10] = dotO
        return dolfin.as_vector(F_expressions)

    def num_states(self):
        "Return number of state variables."
        return 11
