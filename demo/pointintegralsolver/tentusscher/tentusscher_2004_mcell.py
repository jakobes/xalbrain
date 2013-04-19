def init_values(**values):
    """
    Init values
    """
    # Imports
    import dolfin
    from modelparameters.utils import Range

    # Init values
    # Xr1=0, Xr2=1, Xs=0, m=0, h=0.75, j=0.75, d=0, f=1, fCa=1, s=1, r=0,
    # Ca_SR=0.2, Ca_i=0.0002, g=1, Na_i=11.6, V=-86.2, K_i=138.3
    init_values = [0, 1, 0, 0, 0.75, 0.75, 0, 1, 1, 1, 0, 0.2, 0.0002, 1,\
        11.6, -86.2, 138.3]

    # State indices and limit checker
    state_ind = dict(Xr1=(0, Range()), Xr2=(1, Range()), Xs=(2, Range()),\
        m=(3, Range()), h=(4, Range()), j=(5, Range()), d=(6, Range()), f=(7,\
        Range()), fCa=(8, Range()), s=(9, Range()), r=(10, Range()),\
        Ca_SR=(11, Range()), Ca_i=(12, Range()), g=(13, Range()), Na_i=(14,\
        Range()), V=(15, Range()), K_i=(16, Range()))

    for state_name, value in values.items():
        if state_name not in state_ind:
            raise ValueError("{{0}} is not a state.".format(state_name))
        ind, range = state_ind[state_name]
        if value not in range:
            raise ValueError("While setting '{0}' {1}".format(state_name,\
                range.format_not_in(value)))

        # Assign value
        init_values[ind] = value
    init_values = dolfin.Constant(tuple(init_values))

    return init_values

def default_parameters(**values):
    """
    Parameter values
    """
    # Imports
    import dolfin
    from modelparameters.utils import Range

    # Param values
    # P_kna=0.03, g_K1=5.405, g_Kr=0.096, g_Ks=0.062, g_Na=14.838,
    # g_bna=0.00029, g_CaL=0.000175, g_bca=0.000592, g_to=0.294,
    # K_mNa=40, K_mk=1, P_NaK=1.362, K_NaCa=1000, K_sat=0.1,
    # Km_Ca=1.38, Km_Nai=87.5, alpha=2.5, gamma=0.35, K_pCa=0.0005,
    # g_pCa=0.825, g_pK=0.0146, Buf_c=0.15, Buf_sr=10, Ca_o=2,
    # K_buf_c=0.001, K_buf_sr=0.3, K_up=0.00025, V_leak=8e-05,
    # V_sr=0.001094, Vmax_up=0.000425, a_rel=0.016464, b_rel=0.25,
    # c_rel=0.008232, tau_g=2, Na_o=140, Cm=0.185, F=96485.3415,
    # R=8314.472, T=310, V_c=0.016404, stim_amplitude=52,
    # stim_duration=1, stim_start=5, K_o=5.4
    param_values = [0.03, 5.405, 0.096, 0.062, 14.838, 0.00029, 0.000175,\
        0.000592, 0.294, 40, 1, 1.362, 1000, 0.1, 1.38, 87.5, 2.5, 0.35,\
        0.0005, 0.825, 0.0146, 0.15, 10, 2, 0.001, 0.3, 0.00025, 8e-05,\
        0.001094, 0.000425, 0.016464, 0.25, 0.008232, 2, 140, 0.185,\
        96485.3415, 8314.472, 310, 0.016404, 52, 1, 5, 5.4]

    # Parameter indices and limit checker
    param_ind = dict(P_kna=(0, Range()), g_K1=(1, Range()), g_Kr=(2,\
        Range()), g_Ks=(3, Range()), g_Na=(4, Range()), g_bna=(5, Range()),\
        g_CaL=(6, Range()), g_bca=(7, Range()), g_to=(8, Range()), K_mNa=(9,\
        Range()), K_mk=(10, Range()), P_NaK=(11, Range()), K_NaCa=(12,\
        Range()), K_sat=(13, Range()), Km_Ca=(14, Range()), Km_Nai=(15,\
        Range()), alpha=(16, Range()), gamma=(17, Range()), K_pCa=(18,\
        Range()), g_pCa=(19, Range()), g_pK=(20, Range()), Buf_c=(21,\
        Range()), Buf_sr=(22, Range()), Ca_o=(23, Range()), K_buf_c=(24,\
        Range()), K_buf_sr=(25, Range()), K_up=(26, Range()), V_leak=(27,\
        Range()), V_sr=(28, Range()), Vmax_up=(29, Range()), a_rel=(30,\
        Range()), b_rel=(31, Range()), c_rel=(32, Range()), tau_g=(33,\
        Range()), Na_o=(34, Range()), Cm=(35, Range()), F=(36, Range()),\
        R=(37, Range()), T=(38, Range()), V_c=(39, Range()),\
        stim_amplitude=(40, Range()), stim_duration=(41, Range()),\
        stim_start=(42, Range()), K_o=(43, Range()))

    for param_name, value in values.items():
        if param_name not in param_ind:
            raise ValueError("{{0}} is not a param".format(param_name))
        ind, range = param_ind[param_name]
        if value not in range:
            raise ValueError("While setting '{0}' {1}".format(param_name,\
                range.format_not_in(value)))

        # Assign value
        param_values[ind] = value
    param_values = dolfin.Constant(tuple(param_values))

    return param_values

def rhs(states, time, parameters, dy=None):
    """
    Calculate right hand side
    """
    # Imports
    import ufl
    import dolfin

    # Assign states
    assert(isinstance(states, dolfin.Function))
    assert(states.function_space().depth() == 1)
    assert(states.function_space().num_sub_spaces() == 17)
    Xr1, Xr2, Xs, m, h, j, d, f, fCa, s, r, Ca_SR, Ca_i, g, Na_i, V, K_i =\
        dolfin.split(states)

    # Assign parameters
    assert(isinstance(parameters, (dolfin.Function, dolfin.Constant)))
    if isinstance(parameters, dolfin.Function):
        assert(parameters.function_space().depth() == 1)
        assert(parameters.function_space().num_sub_spaces() == 44)
    else:
        assert(parameters.value_size() == 44)
    P_kna, g_K1, g_Kr, g_Ks, g_Na, g_bna, g_CaL, g_bca, g_to, K_mNa, K_mk,\
        P_NaK, K_NaCa, K_sat, Km_Ca, Km_Nai, alpha, gamma, K_pCa, g_pCa,\
        g_pK, Buf_c, Buf_sr, Ca_o, K_buf_c, K_buf_sr, K_up, V_leak, V_sr,\
        Vmax_up, a_rel, b_rel, c_rel, tau_g, Na_o, Cm, F, R, T, V_c,\
        stim_amplitude, stim_duration, stim_start, K_o =\
        dolfin.split(parameters)

    # Init test function
    _v = dolfin.TestFunction(states.function_space())

    # Derivative for state Xr1
    dy = (0.00037037037037037*(1.0 +\
        13.5813245225782*ufl.exp(0.0869565217391304*V))*(1.0 + ufl.exp(-9/2 -\
        V/10.0))*(-Xr1 + 1.0/(1.0 +\
        0.0243728440732796*ufl.exp(-0.142857142857143*V))))*_v[0]

    # Derivative for state Xr2
    dy += (0.297619047619048*(1.0 + 0.0497870683678639*ufl.exp(0.05*V))*(1.0 +\
        0.0497870683678639*ufl.exp(-0.05*V))*(-Xr2 + 1.0/(1.0 +\
        39.1212839981532*ufl.exp(0.0416666666666667*V))))*_v[1]

    # Derivative for state Xs
    dy += (0.000909090909090909*ufl.sqrt(1.0 +\
        0.188875602837562*ufl.exp(-0.166666666666667*V))*(1.0 +\
        0.0497870683678639*ufl.exp(0.05*V))*(-Xs + 1.0/(1.0 +\
        0.69967253737513*ufl.exp(-0.0714285714285714*V))))*_v[2]

    # Derivative for state m
    dy += ((1.0 + ufl.exp(-12.0 - V/5.0))*(1.0/((1.0 +\
        0.00184221158116513*ufl.exp(-0.110741971207087*V))*(1.0 +\
        0.00184221158116513*ufl.exp(-0.110741971207087*V))) - m)/(0.1/(1.0 +\
        0.778800783071405*ufl.exp(0.005*V)) + 0.1/(1.0 + ufl.exp(7.0 +\
        V/5.0))))*_v[3]

    # Derivative for state h
    dy += ((-h + 1.0/((1.0 +\
        15212.5932856544*ufl.exp(0.134589502018843*V))*(1.0 +\
        15212.5932856544*ufl.exp(0.134589502018843*V))))*(ufl.conditional(ufl.lt(V,\
        -40.0), 310000.0*ufl.exp(0.3485*V) + 2.7*ufl.exp(0.079*V), 0.77/(0.13 +\
        0.0497581410839387*ufl.exp(-0.0900900900900901*V))) +\
        ufl.conditional(ufl.lt(V, -40.0),\
        4.43126792958051e-7*ufl.exp(-0.147058823529412*V), 0.0)))*_v[4]

    # Derivative for state j
    dy += ((1.0/((1.0 + 15212.5932856544*ufl.exp(0.134589502018843*V))*(1.0 +\
        15212.5932856544*ufl.exp(0.134589502018843*V))) -\
        j)*(ufl.conditional(ufl.lt(V, -40.0),\
        0.02424*ufl.exp(-0.01052*V)/(1.0 +\
        0.00396086833990426*ufl.exp(-0.1378*V)), 0.6*ufl.exp(0.057*V)/(1.0 +\
        0.0407622039783662*ufl.exp(-0.1*V))) + ufl.conditional(ufl.lt(V,\
        -40.0), (37.78 + V)*(-6.948*ufl.exp(-0.04391*V) -\
        25428.0*ufl.exp(0.2444*V))/(1.0 + 50262745825.954*ufl.exp(0.311*V)),\
        0.0)))*_v[5]

    # Derivative for state d
    dy += ((1.0/(1.0 + 0.513417119032592*ufl.exp(-0.133333333333333*V)) -\
        d)/(1.0/(1.0 + 12.1824939607035*ufl.exp(-0.05*V)) + 1.4*(0.25 +\
        1.4/(1.0 + 0.0677244716592409*ufl.exp(-0.0769230769230769*V)))/(1.0 +\
        ufl.exp(1.0 + V/5.0))))*_v[6]

    # Derivative for state f
    dy += ((1.0/(1.0 + 17.4117080633276*ufl.exp(0.142857142857143*V)) -\
        f)/(80.0 + 165.0/(1.0 + ufl.exp(5/2 - V/10.0)) +\
        1125.0*ufl.exp(-0.00416666666666667*((27.0 + V)*(27.0 + V)))))*_v[7]

    # Derivative for state fCa
    dy += (ufl.conditional(ufl.And(ufl.gt(0.157534246575342 +\
        0.684931506849315/(1.0 + 8.03402376701711e+27*ufl.elem_pow(Ca_i,\
        8.0)) + 0.136986301369863/(1.0 +\
        0.391605626676799*ufl.exp(1250.0*Ca_i)) + 0.0684931506849315/(1.0 +\
        0.00673794699908547*ufl.exp(10000.0*Ca_i)), fCa), ufl.gt(V, -60.0)),\
        0.0, 0.0787671232876712 + 0.0342465753424658/(1.0 +\
        0.00673794699908547*ufl.exp(10000.0*Ca_i)) + 0.342465753424658/(1.0 +\
        8.03402376701711e+27*ufl.elem_pow(Ca_i, 8.0)) +\
        0.0684931506849315/(1.0 + 0.391605626676799*ufl.exp(1250.0*Ca_i)) -\
        fCa/2.0))*_v[8]

    # Derivative for state s
    dy += ((-s + 1.0/(1.0 + ufl.exp(4.0 + V/5.0)))/(3.0 +\
        85.0*ufl.exp(-0.003125*((45.0 + V)*(45.0 + V))) + 5.0/(1.0 +\
        ufl.exp(-4.0 + V/5.0))))*_v[9]

    # Derivative for state r
    dy += ((-r + 1.0/(1.0 +\
        28.0316248945261*ufl.exp(-0.166666666666667*V)))/(0.8 +\
        9.5*ufl.exp(-0.000555555555555556*((40.0 + V)*(40.0 + V)))))*_v[10]

    # Derivative for state Ca_SR
    dy += ((Vmax_up/(1.0 + (K_up*K_up)/(Ca_i*Ca_i)) - (-Ca_i + Ca_SR)*V_leak\
        - ((Ca_SR*Ca_SR)*a_rel/((Ca_SR*Ca_SR) + (b_rel*b_rel)) +\
        c_rel)*d*g)*V_c/((1.0 + Buf_sr*K_buf_sr/((K_buf_sr + Ca_SR)*(K_buf_sr\
        + Ca_SR)))*V_sr))*_v[11]

    # Derivative for state Ca_i
    dy += (((-Ca_i + Ca_SR)*V_leak - Vmax_up/(1.0 + (K_up*K_up)/(Ca_i*Ca_i))\
        + ((Ca_SR*Ca_SR)*a_rel/((Ca_SR*Ca_SR) + (b_rel*b_rel)) + c_rel)*d*g -\
        ((V - 0.5*R*T*ufl.ln(Ca_o/Ca_i)/F)*g_bca + Ca_i*g_pCa/(K_pCa + Ca_i)\
        - 2.0*(-(Na_o*Na_o*Na_o)*Ca_i*alpha*ufl.exp((-1.0 + gamma)*F*V/(R*T))\
        + (Na_i*Na_i*Na_i)*Ca_o*ufl.exp(F*V*gamma/(R*T)))*K_NaCa/((1.0 +\
        K_sat*ufl.exp((-1.0 + gamma)*F*V/(R*T)))*((Na_o*Na_o*Na_o) +\
        (Km_Nai*Km_Nai*Km_Nai))*(Km_Ca + Ca_o)) + 4.0*(F*F)*(-0.341*Ca_o +\
        Ca_i*ufl.exp(2.0*F*V/(R*T)))*V*d*f*fCa*g_CaL/((-1.0 +\
        ufl.exp(2.0*F*V/(R*T)))*R*T))*Cm/(2.0*F*V_c))/(1.0 +\
        Buf_c*K_buf_c/((K_buf_c + Ca_i)*(K_buf_c + Ca_i))))*_v[12]

    # Derivative for state g
    dy += (ufl.conditional(ufl.And(ufl.gt(ufl.conditional(ufl.lt(Ca_i,\
        0.00035), 1.0/(1.0 + 5.43991024148102e+20*ufl.elem_pow(Ca_i, 6.0)),\
        1.0/(1.0 + 1.97201988740492e+55*ufl.elem_pow(Ca_i, 16.0))), g),\
        ufl.gt(V, -60.0)), 0.0, (-g + ufl.conditional(ufl.lt(Ca_i, 0.00035),\
        1.0/(1.0 + 5.43991024148102e+20*ufl.elem_pow(Ca_i, 6.0)), 1.0/(1.0 +\
        1.97201988740492e+55*ufl.elem_pow(Ca_i, 16.0))))/tau_g))*_v[13]

    # Derivative for state Na_i
    dy += ((-3.0*(-(Na_o*Na_o*Na_o)*Ca_i*alpha*ufl.exp((-1.0 +\
        gamma)*F*V/(R*T)) +\
        (Na_i*Na_i*Na_i)*Ca_o*ufl.exp(F*V*gamma/(R*T)))*K_NaCa/((1.0 +\
        K_sat*ufl.exp((-1.0 + gamma)*F*V/(R*T)))*((Na_o*Na_o*Na_o) +\
        (Km_Nai*Km_Nai*Km_Nai))*(Km_Ca + Ca_o)) - 3.0*K_o*Na_i*P_NaK/((K_mk +\
        K_o)*(Na_i + K_mNa)*(1.0 + 0.0353*ufl.exp(-F*V/(R*T)) +\
        0.1245*ufl.exp(-0.1*F*V/(R*T)))) - (m*m*m)*(-R*T*ufl.ln(Na_o/Na_i)/F\
        + V)*g_Na*h*j - (-R*T*ufl.ln(Na_o/Na_i)/F +\
        V)*g_bna)*Cm/(F*V_c))*_v[14]

    # Derivative for state V
    dy += (-0.0430331482911935*ufl.sqrt(K_o)*(V -\
        R*T*ufl.ln(K_o/K_i)/F)*g_K1/((1.0 +\
        6.14421235332821e-6*ufl.exp(0.06*V -\
        0.06*R*T*ufl.ln(K_o/K_i)/F))*((0.367879441171442*ufl.exp(0.1*V -\
        0.1*R*T*ufl.ln(K_o/K_i)/F) + 3.06060402008027*ufl.exp(0.0002*V -\
        0.0002*R*T*ufl.ln(K_o/K_i)/F))/(1.0 +\
        ufl.exp(0.5*R*T*ufl.ln(K_o/K_i)/F - 0.5*V)) + 0.1/(1.0 +\
        6.14421235332821e-6*ufl.exp(0.06*V - 0.06*R*T*ufl.ln(K_o/K_i)/F)))) -\
        4.0*(F*F)*(-0.341*Ca_o +\
        Ca_i*ufl.exp(2.0*F*V/(R*T)))*V*d*f*fCa*g_CaL/((-1.0 +\
        ufl.exp(2.0*F*V/(R*T)))*R*T) - (m*m*m)*(-R*T*ufl.ln(Na_o/Na_i)/F +\
        V)*g_Na*h*j - Ca_i*g_pCa/(K_pCa + Ca_i) - (-R*T*ufl.ln(Na_o/Na_i)/F +\
        V)*g_bna - (Xs*Xs)*(V - R*T*ufl.ln((Na_o*P_kna + K_o)/(Na_i*P_kna +\
        K_i))/F)*g_Ks - (V - R*T*ufl.ln(K_o/K_i)/F)*g_to*r*s - (V -\
        R*T*ufl.ln(K_o/K_i)/F)*g_pK/(1.0 +\
        65.4052157419383*ufl.exp(-0.167224080267559*V)) -\
        K_o*Na_i*P_NaK/((K_mk + K_o)*(Na_i + K_mNa)*(1.0 +\
        0.0353*ufl.exp(-F*V/(R*T)) + 0.1245*ufl.exp(-0.1*F*V/(R*T)))) -\
        0.430331482911935*ufl.sqrt(K_o)*(V -\
        R*T*ufl.ln(K_o/K_i)/F)*Xr1*Xr2*g_Kr -\
        (-(Na_o*Na_o*Na_o)*Ca_i*alpha*ufl.exp((-1.0 + gamma)*F*V/(R*T)) +\
        (Na_i*Na_i*Na_i)*Ca_o*ufl.exp(F*V*gamma/(R*T)))*K_NaCa/((1.0 +\
        K_sat*ufl.exp((-1.0 + gamma)*F*V/(R*T)))*((Na_o*Na_o*Na_o) +\
        (Km_Nai*Km_Nai*Km_Nai))*(Km_Ca + Ca_o)) -\
        ufl.conditional(ufl.And(ufl.ge(time, stim_start), ufl.le(time,\
        stim_start + stim_duration)), -stim_amplitude, 0.0) - (V -\
        0.5*R*T*ufl.ln(Ca_o/Ca_i)/F)*g_bca)*_v[15]

    # Derivative for state K_i
    dy += ((-0.0430331482911935*ufl.sqrt(K_o)*(V -\
        R*T*ufl.ln(K_o/K_i)/F)*g_K1/((1.0 +\
        6.14421235332821e-6*ufl.exp(0.06*V -\
        0.06*R*T*ufl.ln(K_o/K_i)/F))*((0.367879441171442*ufl.exp(0.1*V -\
        0.1*R*T*ufl.ln(K_o/K_i)/F) + 3.06060402008027*ufl.exp(0.0002*V -\
        0.0002*R*T*ufl.ln(K_o/K_i)/F))/(1.0 +\
        ufl.exp(0.5*R*T*ufl.ln(K_o/K_i)/F - 0.5*V)) + 0.1/(1.0 +\
        6.14421235332821e-6*ufl.exp(0.06*V - 0.06*R*T*ufl.ln(K_o/K_i)/F)))) +\
        2.0*K_o*Na_i*P_NaK/((K_mk + K_o)*(Na_i + K_mNa)*(1.0 +\
        0.0353*ufl.exp(-F*V/(R*T)) + 0.1245*ufl.exp(-0.1*F*V/(R*T)))) -\
        (Xs*Xs)*(V - R*T*ufl.ln((Na_o*P_kna + K_o)/(Na_i*P_kna +\
        K_i))/F)*g_Ks - (V - R*T*ufl.ln(K_o/K_i)/F)*g_to*r*s - (V -\
        R*T*ufl.ln(K_o/K_i)/F)*g_pK/(1.0 +\
        65.4052157419383*ufl.exp(-0.167224080267559*V)) -\
        0.430331482911935*ufl.sqrt(K_o)*(V -\
        R*T*ufl.ln(K_o/K_i)/F)*Xr1*Xr2*g_Kr -\
        ufl.conditional(ufl.And(ufl.ge(time, stim_start), ufl.le(time,\
        stim_start + stim_duration)), -stim_amplitude,\
        0.0))*Cm/(F*V_c))*_v[16]

    # Return dy
    return dy

