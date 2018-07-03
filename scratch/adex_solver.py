r"""Solve the equation
.. math::

    C\frac{d V}{d t} = -g_L(V - E_L) + g_L\Delta_T\exp{\frac{V - V_T}{\Delta_T}}
                            - w - I
    \tau_w\frac{d w}{d t} = a(V - E_L} - w

    at \quad t = t^f \quad reset \quad V \leftarrow V_r
    at t = t^f \quad reset w \leftarrow w + b
"""


import numpy as np



class Adex:
    r"""Class representing the adex cell model with some default parameters.
    The equation will be written on the form

    .. math::
        \frac{d V}{d t} = I(V, w}
        \frac{d w}{ dt} = F(V, w}
    """
    def __init__(self, params):
        parameters = Adex.default_parameters()
        parameters.update(params)
        self._params = parameters

    @staticmethod
    def default_parameters():
        params = {"C": 281,           # Membrane capacitance (pF)
                  "g_L": 30,          # Leak conductance (nS)
                  "E_L": -70.6,       # Leak reversal potential (mV)
                  "V_T": -50.4,       # Spike threshold (mV)
                  "Delta_T": 2,       # Slope factor (mV)
                  "tau_w": 144,       # Adaptation time constant (ms)
                  "a": 4,             # Subthreshold adaptation (nS)
                  "spike": 20,        # When to reset (mV)
                  "b": 0.0805         # Spike-triggered adaptation (nA)
                  }
        return params

    def I(self, V, w):
        """Right hand side of the ODE for the transmembrane potential."""
        g_L = self._params["g_L"]
        E_L = self._params["E_L"]
        Dt = self._params["Delta_T"]
        C = self._params["C"]
        V_T = self._params["V_T"]

        return 1./C*(g_L*Dt*np.exp((V - V_T)/Dt) - w - g_L*(V - E_L))

    def F(self, V, w):
        """Right hand side of the ODE for the adaptation."""
        tau_w = self._params["tau_w"]
        a = self._params["a"]
        E_L = self._params["E_L"]

        return 1./tau_w*(a*(V - E_L) - w)

    def update(self, V, w):
        """Update V, w if the model is spiking. 

        Returns:
            V, w (float, float): if vectorize is False
        """
        spike = self._params["spike"]
        b = self._params["b"]
        E_L = self._params["E_L"]

        if V > spike:
            V = E_L
            w += b
        return V, w


def solve(t0, T, dt, adex_model, ic, stimulus):
    """1st order forward difference scheme to solve the adex model.
    """
    time = np.arange(t0, T, dt)     # for plotting and time-stepping purposes
    V = np.zeros_like(time)
    w = np.zeros_like(time)

    # ICs
    V[0] = ic[0]
    w[0] = ic[1]

    I = adex.I
    F = adex.F

    for i, t in enumerate(time[1:]):
        V_tmp = V[i] + dt*I(V[i], w[i]) + dt*stimulus(t)
        w_tmp = w[i] + dt*F(V[i], w[i])

        # In the fenics implementation, this step requires (at the moment) copying the current
        # solution
        V[i + 1], w[i + 1] = adex.update(V_tmp, w_tmp)      # update if spiking

    return time, V, w


if __name__ == "__main__":
    params = Adex.default_parameters()
    adex = Adex(params)

    stimulus = lambda t: 5*(t < 50)     # stimulus applied to adex.I(V, w)
    time, V, w = solve(0, 100, 5e-4, adex, (params["E_L"], 0), stimulus)

    import matplotlib.pyplot as plt
    import matplotlib as mpl
    # plt.xkcd()

    font = {'size': 40}
    mpl.rc('font', **font)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(time, V, label=r"V")
    ax.plot(time, w, label=r"w")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Transmembrane Potential (mV)")
    plt.legend()
    # plt.show()
    plt.savefig("foo.png")
