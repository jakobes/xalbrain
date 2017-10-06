
"""This module contains a Grandi_pasqualini_bers_2010 cardiac cell model

The module was autogenerated from a gotran ode file
"""
from __future__ import division
from collections import OrderedDict
import ufl

from xalbrain.dolfinimport import *
from xalbrain.cellmodels import CardiacCellModel

class Grandi_pasqualini_bers_2010(CardiacCellModel):
    def __init__(self, params=None, init_conditions=None):
        """
        Create cardiac cell model

        *Arguments*
         params (dict, :py:class:`dolfin.Mesh`, optional)
           optional model parameters
         init_conditions (dict, :py:class:`dolfin.Mesh`, optional)
           optional initial conditions
        """
        CardiacCellModel.__init__(self, params, init_conditions)

    @staticmethod
    def default_parameters():
        "Set-up and return default parameters."
        params = OrderedDict([("Fjunc", 0.11),
                              ("Fjunc_CaL", 0.9),
                              ("cellLength", 100),
                              ("cellRadius", 10.25),
                              ("distJuncSL", 0.5),
                              ("distSLcyto", 0.45),
                              ("junctionLength", 0.16),
                              ("junctionRadius", 0.015),
                              ("GNa", 23),
                              ("GNaB", 0.000597),
                              ("IbarNaK", 1.8),
                              ("KmKo", 1.5),
                              ("KmNaip", 11),
                              ("Q10KmNai", 1.39),
                              ("Q10NaK", 1.63),
                              ("gkp", 0.002),
                              ("pNaK", 0.01833),
                              ("epi", 1),
                              ("GClB", 0.009),
                              ("GClCa", 0.0548125),
                              ("KdClCa", 0.1),
                              ("Q10CaL", 1.8),
                              ("pCa", 0.00027),
                              ("pK", 1.35e-07),
                              ("pNa", 7.5e-09),
                              ("IbarNCX", 4.5),
                              ("Kdact", 0.00015),
                              ("KmCai", 0.00359),
                              ("KmCao", 1.3),
                              ("KmNai", 12.29),
                              ("KmNao", 87.5),
                              ("Q10NCX", 1.57),
                              ("ksat", 0.32),
                              ("nu", 0.27),
                              ("IbarSLCaP", 0.0673),
                              ("KmPCa", 0.0005),
                              ("Q10SLCaP", 2.35),
                              ("GCaB", 0.0005513),
                              ("Kmf", 0.000246),
                              ("Kmr", 1.7),
                              ("MaxSR", 15),
                              ("MinSR", 1),
                              ("Q10SRCaP", 2.6),
                              ("Vmax_SRCaP", 0.0053114),
                              ("ec50SR", 0.45),
                              ("hillSRCaP", 1.787),
                              ("kiCa", 0.5),
                              ("kim", 0.005),
                              ("koCa", 10),
                              ("kom", 0.06),
                              ("ks", 25),
                              ("Bmax_Naj", 7.561),
                              ("Bmax_Nasl", 1.65),
                              ("koff_na", 0.001),
                              ("kon_na", 0.0001),
                              ("Bmax_CaM", 0.024),
                              ("Bmax_SR", 0.0171),
                              ("Bmax_TnChigh", 0.14),
                              ("Bmax_TnClow", 0.07),
                              ("Bmax_myosin", 0.14),
                              ("koff_cam", 0.238),
                              ("koff_myoca", 0.00046),
                              ("koff_myomg", 5.7e-05),
                              ("koff_sr", 0.06),
                              ("koff_tnchca", 3.2e-05),
                              ("koff_tnchmg", 0.00333),
                              ("koff_tncl", 0.0196),
                              ("kon_cam", 34),
                              ("kon_myoca", 13.8),
                              ("kon_myomg", 0.0157),
                              ("kon_sr", 100),
                              ("kon_tnchca", 2.37),
                              ("kon_tnchmg", 0.003),
                              ("kon_tncl", 32.7),
                              ("Bmax_SLhighj0", 0.000165),
                              ("Bmax_SLhighsl0", 0.0134),
                              ("Bmax_SLlowj0", 0.00046),
                              ("Bmax_SLlowsl0", 0.0374),
                              ("koff_slh", 0.03),
                              ("koff_sll", 1.3),
                              ("kon_slh", 100),
                              ("kon_sll", 100),
                              ("Bmax_Csqn0", 0.14),
                              ("DcaJuncSL", 1.64e-06),
                              ("DcaSLcyto", 1.22e-06),
                              ("J_ca_juncsl", 8.2413e-13),
                              ("J_ca_slmyo", 3.7243e-12),
                              ("koff_csqn", 65),
                              ("kon_csqn", 100),
                              ("DnaJuncSL", 1.09e-05),
                              ("DnaSLcyto", 1.79e-05),
                              ("J_na_juncsl", 1.8313e-14),
                              ("J_na_slmyo", 1.6386e-12),
                              ("Nao", 140),
                              ("Ko", 5.4),
                              ("Cao", 1.8),
                              ("Cli", 15),
                              ("Clo", 150),
                              ("Mgi", 1),
                              ("Cmem", 1.381e-10),
                              ("Frdy", 96485),
                              ("R", 8314),
                              ("Temp", 310),
                              ("stim_amplitude", 0.0),
                              ("stim_duration", 1.0),
                              ("stim_period", 1000.0),
                              ("stim_start", 1.0)])
        return params

    @staticmethod
    def default_initial_conditions():
        "Set-up and return default initial conditions."
        ic = OrderedDict([("V_m", -81.4552030513),
                          ("h", 0.626221949492),
                          ("j", 0.62455357249),
                          ("m", 0.00379308741444),
                          ("x_kr", 0.0210022533039),
                          ("x_ks", 0.00428016666259),
                          ("x_to_f", 0.000440438103759),
                          ("x_to_s", 0.000440445885643),
                          ("y_to_f", 0.999995844039),
                          ("y_to_s", 0.785115828275),
                          ("d", 2.92407183949e-06),
                          ("f", 0.995135796704),
                          ("f_Ca_Bj", 0.0246760872106),
                          ("f_Ca_Bsl", 0.0152723084239),
                          ("Ry_Ri", 9.07666168961e-08),
                          ("Ry_Ro", 7.40481128854e-07),
                          ("Ry_Rr", 0.890806040818),
                          ("Na_Bj", 3.45437733033),
                          ("Na_Bsl", 0.753740951478),
                          ("CaM", 0.000295573424135),
                          ("Myo_c", 0.00192322252438),
                          ("Myo_m", 0.137560495023),
                          ("SRB", 0.00217360235649),
                          ("Tn_CHc", 0.117412025937),
                          ("Tn_CHm", 0.0106160166693),
                          ("Tn_CL", 0.00893455096919),
                          ("SLH_j", 0.0735890020284),
                          ("SLH_sl", 0.114583623437),
                          ("SLL_j", 0.0074052452168),
                          ("SLL_sl", 0.00990339304377),
                          ("Ca_sr", 0.554760499828),
                          ("Csqn_b", 1.19723145924),
                          ("Na_i", 8.40513364345),
                          ("Na_j", 8.40537012593),
                          ("Na_sl", 8.40491910001),
                          ("K_i", 120),
                          ("Ca_i", 8.72509677797e-05),
                          ("Ca_j", 0.000175882395147),
                          ("Ca_sl", 0.000106779509977)])
        return ic

    def _I(self, v, s, time):
        """
        Original gotran transmembrane current dV/dt
        """
        time = time if time else Constant(0.0)

        # Assign states
        V_m = v
        assert(len(s) == 38)
        h, j, m, x_kr, x_ks, x_to_f, x_to_s, y_to_f, y_to_s, d, f, f_Ca_Bj,\
            f_Ca_Bsl, Ry_Ri, Ry_Ro, Ry_Rr, Na_Bj, Na_Bsl, CaM, Myo_c, Myo_m,\
            SRB, Tn_CHc, Tn_CHm, Tn_CL, SLH_j, SLH_sl, SLL_j, SLL_sl, Ca_sr,\
            Csqn_b, Na_i, Na_j, Na_sl, K_i, Ca_i, Ca_j, Ca_sl = s

        # Assign parameters
        Fjunc = self._parameters["Fjunc"]
        Fjunc_CaL = self._parameters["Fjunc_CaL"]
        GNa = self._parameters["GNa"]
        GNaB = self._parameters["GNaB"]
        IbarNaK = self._parameters["IbarNaK"]
        KmKo = self._parameters["KmKo"]
        KmNaip = self._parameters["KmNaip"]
        gkp = self._parameters["gkp"]
        pNaK = self._parameters["pNaK"]
        epi = self._parameters["epi"]
        GClB = self._parameters["GClB"]
        GClCa = self._parameters["GClCa"]
        KdClCa = self._parameters["KdClCa"]
        Q10CaL = self._parameters["Q10CaL"]
        pCa = self._parameters["pCa"]
        pK = self._parameters["pK"]
        pNa = self._parameters["pNa"]
        IbarNCX = self._parameters["IbarNCX"]
        Kdact = self._parameters["Kdact"]
        KmCai = self._parameters["KmCai"]
        KmCao = self._parameters["KmCao"]
        KmNai = self._parameters["KmNai"]
        KmNao = self._parameters["KmNao"]
        Q10NCX = self._parameters["Q10NCX"]
        ksat = self._parameters["ksat"]
        nu = self._parameters["nu"]
        IbarSLCaP = self._parameters["IbarSLCaP"]
        KmPCa = self._parameters["KmPCa"]
        Q10SLCaP = self._parameters["Q10SLCaP"]
        GCaB = self._parameters["GCaB"]
        Nao = self._parameters["Nao"]
        Ko = self._parameters["Ko"]
        Cao = self._parameters["Cao"]
        Cli = self._parameters["Cli"]
        Clo = self._parameters["Clo"]
        Frdy = self._parameters["Frdy"]
        R = self._parameters["R"]
        Temp = self._parameters["Temp"]

        # Init return args
        current = [ufl.zero()]*1

        # Expressions for the Geometry component
        Fsl = 1 - Fjunc
        Fsl_CaL = 1 - Fjunc_CaL

        # Expressions for the Reversal potentials component
        FoRT = Frdy/(R*Temp)
        ena_junc = ufl.ln(Nao/Na_j)/FoRT
        ena_sl = ufl.ln(Nao/Na_sl)/FoRT
        ek = ufl.ln(Ko/K_i)/FoRT
        eca_junc = ufl.ln(Cao/Ca_j)/(2*FoRT)
        eca_sl = ufl.ln(Cao/Ca_sl)/(2*FoRT)
        ecl = ufl.ln(Cli/Clo)/FoRT
        Qpow = -31 + Temp/10

        # Expressions for the I_Na component
        I_Na_junc = Fjunc*GNa*(m*m*m)*(-ena_junc + V_m)*h*j
        I_Na_sl = GNa*(m*m*m)*(-ena_sl + V_m)*Fsl*h*j

        # Expressions for the I_NaBK component
        I_nabk_junc = Fjunc*GNaB*(-ena_junc + V_m)
        I_nabk_sl = GNaB*(-ena_sl + V_m)*Fsl

        # Expressions for the I_NaK component
        sigma = -1/7 + ufl.exp(0.0148588410104*Nao)/7
        fnak = 1.0/(1 + 0.1245*ufl.exp(-0.1*FoRT*V_m) +\
            0.0365*ufl.exp(-FoRT*V_m)*sigma)
        I_nak_junc = Fjunc*IbarNaK*Ko*fnak/((1 + ufl.elem_pow(KmNaip,\
            4)/ufl.elem_pow(Na_j, 4))*(KmKo + Ko))
        I_nak_sl = IbarNaK*Ko*Fsl*fnak/((1 + ufl.elem_pow(KmNaip,\
            4)/ufl.elem_pow(Na_sl, 4))*(KmKo + Ko))
        I_nak = I_nak_junc + I_nak_sl

        # Expressions for the I_Kr component
        gkr = 0.0150616019019*ufl.sqrt(Ko)
        rkr = 1.0/(1 + ufl.exp(37/12 + V_m/24))
        I_kr = (-ek + V_m)*gkr*rkr*x_kr

        # Expressions for the I_Kp component
        kp_kp = 1.0/(1 + 1786.47556538*ufl.exp(-0.167224080268*V_m))
        I_kp_junc = Fjunc*gkp*(-ek + V_m)*kp_kp
        I_kp_sl = gkp*(-ek + V_m)*Fsl*kp_kp
        I_kp = I_kp_sl + I_kp_junc

        # Expressions for the I_Ks component
        eks = ufl.ln((Nao*pNaK + Ko)/(pNaK*Na_i + K_i))/FoRT
        gks_junc = 0.0035
        gks_sl = 0.0035
        I_ks_junc = Fjunc*gks_junc*(x_ks*x_ks)*(-eks + V_m)
        I_ks_sl = gks_sl*(x_ks*x_ks)*(-eks + V_m)*Fsl
        I_ks = I_ks_sl + I_ks_junc

        # Expressions for the I_to component
        GtoSlow = ufl.conditional(ufl.eq(epi, 1), 0.0156, 0.037596)
        GtoFast = ufl.conditional(ufl.eq(epi, 1), 0.1144, 0.001404)
        I_tos = (-ek + V_m)*GtoSlow*x_to_s*y_to_s
        I_tof = (-ek + V_m)*GtoFast*x_to_f*y_to_f
        I_to = I_tos + I_tof

        # Expressions for the I_Ki component
        aki = 1.02/(1 + 7.35454251046e-07*ufl.exp(0.2385*V_m - 0.2385*ek))
        bki = (0.762624006506*ufl.exp(0.08032*V_m - 0.08032*ek) +\
            1.15340563519e-16*ufl.exp(0.06175*V_m - 0.06175*ek))/(1 +\
            0.0867722941577*ufl.exp(-0.5143*V_m + 0.5143*ek))
        kiss = aki/(aki + bki)
        I_ki = 0.150616019019*ufl.sqrt(Ko)*(-ek + V_m)*kiss

        # Expressions for the I_ClCa component
        I_ClCa_junc = Fjunc*GClCa*(-ecl + V_m)/(1 + KdClCa/Ca_j)
        I_ClCa_sl = GClCa*(-ecl + V_m)*Fsl/(1 + KdClCa/Ca_sl)
        I_ClCa = I_ClCa_sl + I_ClCa_junc
        I_Clbk = GClB*(-ecl + V_m)

        # Expressions for the I_Ca component
        fcaCaMSL = 0
        fcaCaj = 0
        ibarca_j = 4*Frdy*pCa*(-0.341*Cao +\
            0.341*Ca_j*ufl.exp(2*FoRT*V_m))*FoRT*V_m/(-1 +\
            ufl.exp(2*FoRT*V_m))
        ibarca_sl = 4*Frdy*pCa*(-0.341*Cao +\
            0.341*Ca_sl*ufl.exp(2*FoRT*V_m))*FoRT*V_m/(-1 +\
            ufl.exp(2*FoRT*V_m))
        ibark = Frdy*pK*(-0.75*Ko + 0.75*K_i*ufl.exp(FoRT*V_m))*FoRT*V_m/(-1 +\
            ufl.exp(FoRT*V_m))
        ibarna_j = Frdy*pNa*(-0.75*Nao +\
            0.75*Na_j*ufl.exp(FoRT*V_m))*FoRT*V_m/(-1 + ufl.exp(FoRT*V_m))
        ibarna_sl = Frdy*pNa*(0.75*Na_sl*ufl.exp(FoRT*V_m) -\
            0.75*Nao)*FoRT*V_m/(-1 + ufl.exp(FoRT*V_m))
        I_Ca_junc = 0.45*Fjunc_CaL*ufl.elem_pow(Q10CaL, Qpow)*(1 - f_Ca_Bj +\
            fcaCaj)*d*f*ibarca_j
        I_Ca_sl = 0.45*ufl.elem_pow(Q10CaL, Qpow)*(1 - f_Ca_Bsl +\
            fcaCaMSL)*Fsl_CaL*d*f*ibarca_sl
        I_CaK = 0.45*ufl.elem_pow(Q10CaL, Qpow)*(Fjunc_CaL*(1 - f_Ca_Bj +\
            fcaCaj) + (1 - f_Ca_Bsl + fcaCaMSL)*Fsl_CaL)*d*f*ibark
        I_CaNa_junc = 0.45*Fjunc_CaL*ufl.elem_pow(Q10CaL, Qpow)*(1 - f_Ca_Bj\
            + fcaCaj)*d*f*ibarna_j
        I_CaNa_sl = 0.45*ufl.elem_pow(Q10CaL, Qpow)*(1 - f_Ca_Bsl +\
            fcaCaMSL)*Fsl_CaL*d*f*ibarna_sl

        # Expressions for the I_NCX component
        Ka_junc = 1.0/(1 + (Kdact*Kdact)/(Ca_j*Ca_j))
        Ka_sl = 1.0/(1 + (Kdact*Kdact)/(Ca_sl*Ca_sl))
        s1_junc = Cao*(Na_j*Na_j*Na_j)*ufl.exp(nu*FoRT*V_m)
        s1_sl = Cao*(Na_sl*Na_sl*Na_sl)*ufl.exp(nu*FoRT*V_m)
        s2_junc = (Nao*Nao*Nao)*Ca_j*ufl.exp((-1 + nu)*FoRT*V_m)
        s3_junc = KmCao*(Na_j*Na_j*Na_j) + (Nao*Nao*Nao)*Ca_j +\
            Cao*(Na_j*Na_j*Na_j) + KmCai*(Nao*Nao*Nao)*(1 +\
            (Na_j*Na_j*Na_j)/(KmNai*KmNai*KmNai)) + (KmNao*KmNao*KmNao)*(1 +\
            Ca_j/KmCai)*Ca_j
        s2_sl = (Nao*Nao*Nao)*Ca_sl*ufl.exp((-1 + nu)*FoRT*V_m)
        s3_sl = KmCai*(Nao*Nao*Nao)*(1 +\
            (Na_sl*Na_sl*Na_sl)/(KmNai*KmNai*KmNai)) + (Nao*Nao*Nao)*Ca_sl +\
            (KmNao*KmNao*KmNao)*(1 + Ca_sl/KmCai)*Ca_sl +\
            Cao*(Na_sl*Na_sl*Na_sl) + KmCao*(Na_sl*Na_sl*Na_sl)
        I_ncx_junc = Fjunc*IbarNCX*ufl.elem_pow(Q10NCX, Qpow)*(-s2_junc +\
            s1_junc)*Ka_junc/((1 + ksat*ufl.exp((-1 + nu)*FoRT*V_m))*s3_junc)
        I_ncx_sl = IbarNCX*ufl.elem_pow(Q10NCX, Qpow)*(-s2_sl +\
            s1_sl)*Fsl*Ka_sl/((1 + ksat*ufl.exp((-1 + nu)*FoRT*V_m))*s3_sl)

        # Expressions for the I_PCa component
        I_pca_junc = Fjunc*IbarSLCaP*ufl.elem_pow(Q10SLCaP,\
            Qpow)*ufl.elem_pow(Ca_j, 1.6)/(ufl.elem_pow(Ca_j, 1.6) +\
            ufl.elem_pow(KmPCa, 1.6))
        I_pca_sl = IbarSLCaP*ufl.elem_pow(Q10SLCaP, Qpow)*ufl.elem_pow(Ca_sl,\
            1.6)*Fsl/(ufl.elem_pow(Ca_sl, 1.6) + ufl.elem_pow(KmPCa, 1.6))

        # Expressions for the I_CaBK component
        I_cabk_junc = Fjunc*GCaB*(-eca_junc + V_m)
        I_cabk_sl = GCaB*(-eca_sl + V_m)*Fsl

        # Expressions for the Na Concentrations component
        I_Na_tot_junc = 3*I_nak_junc + 3*I_ncx_junc + I_CaNa_junc + I_Na_junc\
            + I_nabk_junc
        I_Na_tot_sl = I_Na_sl + I_nabk_sl + 3*I_nak_sl + I_CaNa_sl + 3*I_ncx_sl

        # Expressions for the K Concentration component
        I_K_tot = -2*I_nak + I_ks + I_CaK + I_kr + I_kp + I_ki + I_to

        # Expressions for the Ca Concentrations component
        I_Ca_tot_junc = I_pca_junc + I_cabk_junc + I_Ca_junc - 2*I_ncx_junc
        I_Ca_tot_sl = -2*I_ncx_sl + I_pca_sl + I_cabk_sl + I_Ca_sl

        # Expressions for the Membrane potential component
        i_Stim = 0
        I_Na_tot = I_Na_tot_junc + I_Na_tot_sl
        I_Cl_tot = I_Clbk + I_ClCa
        I_Ca_tot = I_Ca_tot_junc + I_Ca_tot_sl
        I_tot = I_Na_tot + I_K_tot + I_Cl_tot + I_Ca_tot
        current[0] = -I_tot - i_Stim

        # Return results
        return current[0]

    def I(self, v, s, time=None):
        """
        Transmembrane current

           I = -dV/dt

        """
        return -self._I(v, s, time)

    def F(self, v, s, time=None):
        """
        Right hand side for ODE system
        """
        time = time if time else Constant(0.0)

        # Assign states
        V_m = v
        assert(len(s) == 38)
        h, j, m, x_kr, x_ks, x_to_f, x_to_s, y_to_f, y_to_s, d, f, f_Ca_Bj,\
            f_Ca_Bsl, Ry_Ri, Ry_Ro, Ry_Rr, Na_Bj, Na_Bsl, CaM, Myo_c, Myo_m,\
            SRB, Tn_CHc, Tn_CHm, Tn_CL, SLH_j, SLH_sl, SLL_j, SLL_sl, Ca_sr,\
            Csqn_b, Na_i, Na_j, Na_sl, K_i, Ca_i, Ca_j, Ca_sl = s

        # Assign parameters
        Fjunc = self._parameters["Fjunc"]
        Fjunc_CaL = self._parameters["Fjunc_CaL"]
        cellLength = self._parameters["cellLength"]
        cellRadius = self._parameters["cellRadius"]
        GNa = self._parameters["GNa"]
        GNaB = self._parameters["GNaB"]
        IbarNaK = self._parameters["IbarNaK"]
        KmKo = self._parameters["KmKo"]
        KmNaip = self._parameters["KmNaip"]
        Q10CaL = self._parameters["Q10CaL"]
        pCa = self._parameters["pCa"]
        pNa = self._parameters["pNa"]
        IbarNCX = self._parameters["IbarNCX"]
        Kdact = self._parameters["Kdact"]
        KmCai = self._parameters["KmCai"]
        KmCao = self._parameters["KmCao"]
        KmNai = self._parameters["KmNai"]
        KmNao = self._parameters["KmNao"]
        Q10NCX = self._parameters["Q10NCX"]
        ksat = self._parameters["ksat"]
        nu = self._parameters["nu"]
        IbarSLCaP = self._parameters["IbarSLCaP"]
        KmPCa = self._parameters["KmPCa"]
        Q10SLCaP = self._parameters["Q10SLCaP"]
        GCaB = self._parameters["GCaB"]
        Kmf = self._parameters["Kmf"]
        Kmr = self._parameters["Kmr"]
        MaxSR = self._parameters["MaxSR"]
        MinSR = self._parameters["MinSR"]
        Q10SRCaP = self._parameters["Q10SRCaP"]
        Vmax_SRCaP = self._parameters["Vmax_SRCaP"]
        ec50SR = self._parameters["ec50SR"]
        hillSRCaP = self._parameters["hillSRCaP"]
        kiCa = self._parameters["kiCa"]
        kim = self._parameters["kim"]
        koCa = self._parameters["koCa"]
        kom = self._parameters["kom"]
        ks = self._parameters["ks"]
        Bmax_Naj = self._parameters["Bmax_Naj"]
        Bmax_Nasl = self._parameters["Bmax_Nasl"]
        koff_na = self._parameters["koff_na"]
        kon_na = self._parameters["kon_na"]
        Bmax_CaM = self._parameters["Bmax_CaM"]
        Bmax_SR = self._parameters["Bmax_SR"]
        Bmax_TnChigh = self._parameters["Bmax_TnChigh"]
        Bmax_TnClow = self._parameters["Bmax_TnClow"]
        Bmax_myosin = self._parameters["Bmax_myosin"]
        koff_cam = self._parameters["koff_cam"]
        koff_myoca = self._parameters["koff_myoca"]
        koff_myomg = self._parameters["koff_myomg"]
        koff_sr = self._parameters["koff_sr"]
        koff_tnchca = self._parameters["koff_tnchca"]
        koff_tnchmg = self._parameters["koff_tnchmg"]
        koff_tncl = self._parameters["koff_tncl"]
        kon_cam = self._parameters["kon_cam"]
        kon_myoca = self._parameters["kon_myoca"]
        kon_myomg = self._parameters["kon_myomg"]
        kon_sr = self._parameters["kon_sr"]
        kon_tnchca = self._parameters["kon_tnchca"]
        kon_tnchmg = self._parameters["kon_tnchmg"]
        kon_tncl = self._parameters["kon_tncl"]
        Bmax_SLhighj0 = self._parameters["Bmax_SLhighj0"]
        Bmax_SLhighsl0 = self._parameters["Bmax_SLhighsl0"]
        Bmax_SLlowj0 = self._parameters["Bmax_SLlowj0"]
        Bmax_SLlowsl0 = self._parameters["Bmax_SLlowsl0"]
        koff_slh = self._parameters["koff_slh"]
        koff_sll = self._parameters["koff_sll"]
        kon_slh = self._parameters["kon_slh"]
        kon_sll = self._parameters["kon_sll"]
        Bmax_Csqn0 = self._parameters["Bmax_Csqn0"]
        J_ca_juncsl = self._parameters["J_ca_juncsl"]
        J_ca_slmyo = self._parameters["J_ca_slmyo"]
        koff_csqn = self._parameters["koff_csqn"]
        kon_csqn = self._parameters["kon_csqn"]
        J_na_juncsl = self._parameters["J_na_juncsl"]
        J_na_slmyo = self._parameters["J_na_slmyo"]
        Nao = self._parameters["Nao"]
        Ko = self._parameters["Ko"]
        Cao = self._parameters["Cao"]
        Mgi = self._parameters["Mgi"]
        Cmem = self._parameters["Cmem"]
        Frdy = self._parameters["Frdy"]
        R = self._parameters["R"]
        Temp = self._parameters["Temp"]

        # Init return args
        F_expressions = [ufl.zero()]*38

        # Expressions for the Geometry component
        Vcell = 1e-15*ufl.pi*cellLength*(cellRadius*cellRadius)
        Vmyo = 0.65*Vcell
        Vsr = 0.035*Vcell
        Vsl = 0.02*Vcell
        Vjunc = 0.000539*Vcell
        Fsl = 1 - Fjunc
        Fsl_CaL = 1 - Fjunc_CaL

        # Expressions for the Reversal potentials component
        FoRT = Frdy/(R*Temp)
        ena_junc = ufl.ln(Nao/Na_j)/FoRT
        ena_sl = ufl.ln(Nao/Na_sl)/FoRT
        eca_junc = ufl.ln(Cao/Ca_j)/(2*FoRT)
        eca_sl = ufl.ln(Cao/Ca_sl)/(2*FoRT)
        Qpow = -31 + Temp/10

        # Expressions for the I_Na component
        mss = 1.0/((1 + 0.00184221158117*ufl.exp(-0.110741971207*V_m))*(1 +\
            0.00184221158117*ufl.exp(-0.110741971207*V_m)))
        taum = 0.1292*ufl.exp(-((2.94658944659 +\
            0.0643500643501*V_m)*(2.94658944659 + 0.0643500643501*V_m))) +\
            0.06487*ufl.exp(-((-0.0943466353678 +\
            0.0195618153365*V_m)*(-0.0943466353678 + 0.0195618153365*V_m)))
        ah = ufl.conditional(ufl.ge(V_m, -40), 0,\
            4.43126792958e-07*ufl.exp(-0.147058823529*V_m))
        bh = ufl.conditional(ufl.ge(V_m, -40), 0.77/(0.13 +\
            0.0497581410839*ufl.exp(-0.0900900900901*V_m)),\
            310000.0*ufl.exp(0.3485*V_m) + 2.7*ufl.exp(0.079*V_m))
        tauh = 1.0/(bh + ah)
        hss = 1.0/((1 + 15212.5932857*ufl.exp(0.134589502019*V_m))*(1 +\
            15212.5932857*ufl.exp(0.134589502019*V_m)))
        aj = ufl.conditional(ufl.ge(V_m, -40), 0, (37.78 +\
            V_m)*(-25428.0*ufl.exp(0.2444*V_m) -\
            6.948e-06*ufl.exp(-0.04391*V_m))/(1 +\
            50262745826.0*ufl.exp(0.311*V_m)))
        bj = ufl.conditional(ufl.ge(V_m, -40), 0.6*ufl.exp(0.057*V_m)/(1 +\
            0.0407622039784*ufl.exp(-0.1*V_m)),\
            0.02424*ufl.exp(-0.01052*V_m)/(1 +\
            0.0039608683399*ufl.exp(-0.1378*V_m)))
        tauj = 1.0/(bj + aj)
        jss = 1.0/((1 + 15212.5932857*ufl.exp(0.134589502019*V_m))*(1 +\
            15212.5932857*ufl.exp(0.134589502019*V_m)))
        F_expressions[2] = (-m + mss)/taum
        F_expressions[0] = (hss - h)/tauh
        F_expressions[1] = (-j + jss)/tauj
        I_Na_junc = Fjunc*GNa*(m*m*m)*(-ena_junc + V_m)*h*j
        I_Na_sl = GNa*(m*m*m)*(-ena_sl + V_m)*Fsl*h*j

        # Expressions for the I_NaBK component
        I_nabk_junc = Fjunc*GNaB*(-ena_junc + V_m)
        I_nabk_sl = GNaB*(-ena_sl + V_m)*Fsl

        # Expressions for the I_NaK component
        sigma = -1/7 + ufl.exp(0.0148588410104*Nao)/7
        fnak = 1.0/(1 + 0.1245*ufl.exp(-0.1*FoRT*V_m) +\
            0.0365*ufl.exp(-FoRT*V_m)*sigma)
        I_nak_junc = Fjunc*IbarNaK*Ko*fnak/((1 + ufl.elem_pow(KmNaip,\
            4)/ufl.elem_pow(Na_j, 4))*(KmKo + Ko))
        I_nak_sl = IbarNaK*Ko*Fsl*fnak/((1 + ufl.elem_pow(KmNaip,\
            4)/ufl.elem_pow(Na_sl, 4))*(KmKo + Ko))

        # Expressions for the I_Kr component
        xrss = 1.0/(1 + ufl.exp(-2 - V_m/5))
        tauxr = 230/(1 + ufl.exp(2 + V_m/20)) + 3300/((1 + ufl.exp(-22/9 -\
            V_m/9))*(1 + ufl.exp(11/9 + V_m/9)))
        F_expressions[3] = (-x_kr + xrss)/tauxr

        # Expressions for the I_Ks component
        xsss = 1.0/(1 + 0.765928338365*ufl.exp(-0.0701754385965*V_m))
        tauxs = 990.1/(1 + 0.841540408868*ufl.exp(-0.070821529745*V_m))
        F_expressions[4] = (-x_ks + xsss)/tauxs

        # Expressions for the I_to component
        xtoss = 1.0/(1 + ufl.exp(19/13 - V_m/13))
        ytoss = 1.0/(1 + 49.4024491055*ufl.exp(V_m/5))
        tauxtos = 0.5 + 9/(1 + ufl.exp(1/5 + V_m/15))
        tauytos = 30 + 800/(1 + ufl.exp(6 + V_m/10))
        F_expressions[6] = (-x_to_s + xtoss)/tauxtos
        F_expressions[8] = (ytoss - y_to_s)/tauytos
        tauxtof = 0.5 + 8.5*ufl.exp(-((9/10 + V_m/50)*(9/10 + V_m/50)))
        tauytof = 7 + 85*ufl.exp(-((40 + V_m)*(40 + V_m))/220)
        F_expressions[5] = (xtoss - x_to_f)/tauxtof
        F_expressions[7] = (ytoss - y_to_f)/tauytof

        # Expressions for the I_Ca component
        fss = 0.6/(1 + ufl.exp(5/2 - V_m/20)) + 1.0/(1 + ufl.exp(35/9 + V_m/9))
        dss = 1.0/(1 + ufl.exp(-5/6 - V_m/6))
        taud = (1 - ufl.exp(-5/6 - V_m/6))*dss/(0.175 + 0.035*V_m)
        tauf = 1.0/(0.02 + 0.0197*ufl.exp(-((0.48865 + 0.0337*V_m)*(0.48865 +\
            0.0337*V_m))))
        F_expressions[9] = (-d + dss)/taud
        F_expressions[10] = (fss - f)/tauf
        F_expressions[11] = 1.7*(1 - f_Ca_Bj)*Ca_j - 0.0119*f_Ca_Bj
        F_expressions[12] = -0.0119*f_Ca_Bsl + 1.7*(1 - f_Ca_Bsl)*Ca_sl
        fcaCaMSL = 0
        fcaCaj = 0
        ibarca_j = 4*Frdy*pCa*(-0.341*Cao +\
            0.341*Ca_j*ufl.exp(2*FoRT*V_m))*FoRT*V_m/(-1 +\
            ufl.exp(2*FoRT*V_m))
        ibarca_sl = 4*Frdy*pCa*(-0.341*Cao +\
            0.341*Ca_sl*ufl.exp(2*FoRT*V_m))*FoRT*V_m/(-1 +\
            ufl.exp(2*FoRT*V_m))
        ibarna_j = Frdy*pNa*(-0.75*Nao +\
            0.75*Na_j*ufl.exp(FoRT*V_m))*FoRT*V_m/(-1 + ufl.exp(FoRT*V_m))
        ibarna_sl = Frdy*pNa*(0.75*Na_sl*ufl.exp(FoRT*V_m) -\
            0.75*Nao)*FoRT*V_m/(-1 + ufl.exp(FoRT*V_m))
        I_Ca_junc = 0.45*Fjunc_CaL*ufl.elem_pow(Q10CaL, Qpow)*(1 - f_Ca_Bj +\
            fcaCaj)*d*f*ibarca_j
        I_Ca_sl = 0.45*ufl.elem_pow(Q10CaL, Qpow)*(1 - f_Ca_Bsl +\
            fcaCaMSL)*Fsl_CaL*d*f*ibarca_sl
        I_CaNa_junc = 0.45*Fjunc_CaL*ufl.elem_pow(Q10CaL, Qpow)*(1 - f_Ca_Bj\
            + fcaCaj)*d*f*ibarna_j
        I_CaNa_sl = 0.45*ufl.elem_pow(Q10CaL, Qpow)*(1 - f_Ca_Bsl +\
            fcaCaMSL)*Fsl_CaL*d*f*ibarna_sl

        # Expressions for the I_NCX component
        Ka_junc = 1.0/(1 + (Kdact*Kdact)/(Ca_j*Ca_j))
        Ka_sl = 1.0/(1 + (Kdact*Kdact)/(Ca_sl*Ca_sl))
        s1_junc = Cao*(Na_j*Na_j*Na_j)*ufl.exp(nu*FoRT*V_m)
        s1_sl = Cao*(Na_sl*Na_sl*Na_sl)*ufl.exp(nu*FoRT*V_m)
        s2_junc = (Nao*Nao*Nao)*Ca_j*ufl.exp((-1 + nu)*FoRT*V_m)
        s3_junc = KmCao*(Na_j*Na_j*Na_j) + (Nao*Nao*Nao)*Ca_j +\
            Cao*(Na_j*Na_j*Na_j) + KmCai*(Nao*Nao*Nao)*(1 +\
            (Na_j*Na_j*Na_j)/(KmNai*KmNai*KmNai)) + (KmNao*KmNao*KmNao)*(1 +\
            Ca_j/KmCai)*Ca_j
        s2_sl = (Nao*Nao*Nao)*Ca_sl*ufl.exp((-1 + nu)*FoRT*V_m)
        s3_sl = KmCai*(Nao*Nao*Nao)*(1 +\
            (Na_sl*Na_sl*Na_sl)/(KmNai*KmNai*KmNai)) + (Nao*Nao*Nao)*Ca_sl +\
            (KmNao*KmNao*KmNao)*(1 + Ca_sl/KmCai)*Ca_sl +\
            Cao*(Na_sl*Na_sl*Na_sl) + KmCao*(Na_sl*Na_sl*Na_sl)
        I_ncx_junc = Fjunc*IbarNCX*ufl.elem_pow(Q10NCX, Qpow)*(-s2_junc +\
            s1_junc)*Ka_junc/((1 + ksat*ufl.exp((-1 + nu)*FoRT*V_m))*s3_junc)
        I_ncx_sl = IbarNCX*ufl.elem_pow(Q10NCX, Qpow)*(-s2_sl +\
            s1_sl)*Fsl*Ka_sl/((1 + ksat*ufl.exp((-1 + nu)*FoRT*V_m))*s3_sl)

        # Expressions for the I_PCa component
        I_pca_junc = Fjunc*IbarSLCaP*ufl.elem_pow(Q10SLCaP,\
            Qpow)*ufl.elem_pow(Ca_j, 1.6)/(ufl.elem_pow(Ca_j, 1.6) +\
            ufl.elem_pow(KmPCa, 1.6))
        I_pca_sl = IbarSLCaP*ufl.elem_pow(Q10SLCaP, Qpow)*ufl.elem_pow(Ca_sl,\
            1.6)*Fsl/(ufl.elem_pow(Ca_sl, 1.6) + ufl.elem_pow(KmPCa, 1.6))

        # Expressions for the I_CaBK component
        I_cabk_junc = Fjunc*GCaB*(-eca_junc + V_m)
        I_cabk_sl = GCaB*(-eca_sl + V_m)*Fsl

        # Expressions for the SR Fluxes component
        kCaSR = MaxSR - (-MinSR + MaxSR)/(1 + ufl.elem_pow(ec50SR/Ca_sr, 2.5))
        koSRCa = koCa/kCaSR
        kiSRCa = kiCa*kCaSR
        RI = 1 - Ry_Ro - Ry_Ri - Ry_Rr
        F_expressions[15] = -(Ca_j*Ca_j)*Ry_Rr*koSRCa + kom*Ry_Ro + kim*RI -\
            Ca_j*Ry_Rr*kiSRCa
        F_expressions[14] = -kom*Ry_Ro - Ca_j*Ry_Ro*kiSRCa + kim*Ry_Ri +\
            (Ca_j*Ca_j)*Ry_Rr*koSRCa
        F_expressions[13] = -kim*Ry_Ri + Ca_j*Ry_Ro*kiSRCa - kom*Ry_Ri +\
            (Ca_j*Ca_j)*RI*koSRCa
        J_SRCarel = ks*(Ca_sr - Ca_j)*Ry_Ro
        J_serca = Vmax_SRCaP*ufl.elem_pow(Q10SRCaP,\
            Qpow)*(-ufl.elem_pow(Ca_sr/Kmr, hillSRCaP) +\
            ufl.elem_pow(Ca_i/Kmf, hillSRCaP))/(1 + ufl.elem_pow(Ca_sr/Kmr,\
            hillSRCaP) + ufl.elem_pow(Ca_i/Kmf, hillSRCaP))
        J_SRleak = 5.348e-06*Ca_sr - 5.348e-06*Ca_j

        # Expressions for the Na Buffers component
        F_expressions[16] = -koff_na*Na_Bj + kon_na*(-Na_Bj + Bmax_Naj)*Na_j
        F_expressions[17] = kon_na*(-Na_Bsl + Bmax_Nasl)*Na_sl - koff_na*Na_Bsl

        # Expressions for the Cytosolic Ca Buffers component
        F_expressions[24] = kon_tncl*(Bmax_TnClow - Tn_CL)*Ca_i -\
            koff_tncl*Tn_CL
        F_expressions[22] = -koff_tnchca*Tn_CHc + kon_tnchca*(-Tn_CHc +\
            Bmax_TnChigh - Tn_CHm)*Ca_i
        F_expressions[23] = Mgi*kon_tnchmg*(-Tn_CHc + Bmax_TnChigh - Tn_CHm)\
            - koff_tnchmg*Tn_CHm
        F_expressions[18] = kon_cam*(-CaM + Bmax_CaM)*Ca_i - koff_cam*CaM
        F_expressions[19] = -koff_myoca*Myo_c + kon_myoca*(-Myo_c +\
            Bmax_myosin - Myo_m)*Ca_i
        F_expressions[20] = Mgi*kon_myomg*(-Myo_c + Bmax_myosin - Myo_m) -\
            koff_myomg*Myo_m
        F_expressions[21] = kon_sr*(Bmax_SR - SRB)*Ca_i - koff_sr*SRB
        J_CaB_cytosol = -koff_tnchca*Tn_CHc - koff_myoca*Myo_c +\
            Mgi*kon_myomg*(-Myo_c + Bmax_myosin - Myo_m) +\
            Mgi*kon_tnchmg*(-Tn_CHc + Bmax_TnChigh - Tn_CHm) -\
            koff_tnchmg*Tn_CHm + kon_tncl*(Bmax_TnClow - Tn_CL)*Ca_i +\
            kon_sr*(Bmax_SR - SRB)*Ca_i - koff_myomg*Myo_m + kon_cam*(-CaM +\
            Bmax_CaM)*Ca_i - koff_cam*CaM - koff_tncl*Tn_CL +\
            kon_myoca*(-Myo_c + Bmax_myosin - Myo_m)*Ca_i +\
            kon_tnchca*(-Tn_CHc + Bmax_TnChigh - Tn_CHm)*Ca_i - koff_sr*SRB

        # Expressions for the Junctional and SL Ca Buffers component
        Bmax_SLlowsl = Bmax_SLlowsl0*Vmyo/Vsl
        Bmax_SLlowj = Bmax_SLlowj0*Vmyo/Vjunc
        Bmax_SLhighsl = Bmax_SLhighsl0*Vmyo/Vsl
        Bmax_SLhighj = Bmax_SLhighj0*Vmyo/Vjunc
        F_expressions[27] = kon_sll*(Bmax_SLlowj - SLL_j)*Ca_j - koff_sll*SLL_j
        F_expressions[28] = kon_sll*(-SLL_sl + Bmax_SLlowsl)*Ca_sl -\
            koff_sll*SLL_sl
        F_expressions[25] = kon_slh*(Bmax_SLhighj - SLH_j)*Ca_j -\
            koff_slh*SLH_j
        F_expressions[26] = kon_slh*(-SLH_sl + Bmax_SLhighsl)*Ca_sl -\
            koff_slh*SLH_sl
        J_CaB_junction = kon_slh*(Bmax_SLhighj - SLH_j)*Ca_j +\
            kon_sll*(Bmax_SLlowj - SLL_j)*Ca_j - koff_slh*SLH_j -\
            koff_sll*SLL_j
        J_CaB_sl = kon_sll*(-SLL_sl + Bmax_SLlowsl)*Ca_sl + kon_slh*(-SLH_sl\
            + Bmax_SLhighsl)*Ca_sl - koff_sll*SLL_sl - koff_slh*SLH_sl

        # Expressions for the SR Ca Concentrations component
        Bmax_Csqn = Bmax_Csqn0*Vmyo/Vsr
        F_expressions[30] = -koff_csqn*Csqn_b + kon_csqn*(Bmax_Csqn -\
            Csqn_b)*Ca_sr
        F_expressions[29] = -kon_csqn*(Bmax_Csqn - Csqn_b)*Ca_sr -\
            J_SRleak*Vmyo/Vsr + koff_csqn*Csqn_b - J_SRCarel + J_serca

        # Expressions for the Na Concentrations component
        I_Na_tot_junc = 3*I_nak_junc + 3*I_ncx_junc + I_CaNa_junc + I_Na_junc\
            + I_nabk_junc
        I_Na_tot_sl = I_Na_sl + I_nabk_sl + 3*I_nak_sl + I_CaNa_sl + 3*I_ncx_sl
        F_expressions[32] = -Cmem*I_Na_tot_junc/(Frdy*Vjunc) +\
            J_na_juncsl*(Na_sl - Na_j)/Vjunc - F_expressions[16]
        F_expressions[33] = -F_expressions[17] + J_na_slmyo*(-Na_sl +\
            Na_i)/Vsl + J_na_juncsl*(-Na_sl + Na_j)/Vsl -\
            Cmem*I_Na_tot_sl/(Frdy*Vsl)
        F_expressions[31] = J_na_slmyo*(Na_sl - Na_i)/Vmyo

        # Expressions for the K Concentration component
        F_expressions[34] = Constant(0.0)

        # Expressions for the Ca Concentrations component
        I_Ca_tot_junc = I_pca_junc + I_cabk_junc + I_Ca_junc - 2*I_ncx_junc
        I_Ca_tot_sl = -2*I_ncx_sl + I_pca_sl + I_cabk_sl + I_Ca_sl
        F_expressions[36] = J_ca_juncsl*(Ca_sl - Ca_j)/Vjunc +\
            J_SRCarel*Vsr/Vjunc - J_CaB_junction -\
            Cmem*I_Ca_tot_junc/(2*Frdy*Vjunc) + J_SRleak*Vmyo/Vjunc
        F_expressions[37] = -J_CaB_sl + J_ca_juncsl*(Ca_j - Ca_sl)/Vsl -\
            Cmem*I_Ca_tot_sl/(2*Frdy*Vsl) + J_ca_slmyo*(Ca_i - Ca_sl)/Vsl
        F_expressions[35] = -J_CaB_cytosol + J_ca_slmyo*(Ca_sl - Ca_i)/Vmyo -\
            J_serca*Vsr/Vmyo

        # Return results
        return dolfin.as_vector(F_expressions)

    def num_states(self):
        return 38

    def __str__(self):
        return 'Grandi_pasqualini_bers_2010 cardiac cell model'
