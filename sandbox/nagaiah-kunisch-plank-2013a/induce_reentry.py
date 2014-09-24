from beatadjoint import *

class S1S2Stimulation(Expression):
    "A S1-S2 stimulation protocol."
    def __init__(self, t):
        self.t = t # ms
    # FIXME
    def eval(self, value, x):
        if float(self.t) >= 435 and float(self.t) <= 439:
            value[0] = 100. # mV
        else:
            value[0] = 0.0

def main():

    n = 128
    mesh = RectangleMesh(0.0, 0.0, 2.0, 2.0, n, n)
    time = Constant(0.0)

    M_i = diag(as_vector([2.0e-3, 3.1e-4])) # S/cm
    M_e = diag(as_vector([2.0e-3, 1.3e-3])) # S/cm

    cell_model = RogersMcCulloch()

    I_s = S1S2Stimulation()
    tissue = CardiacModel(mesh, time, M_i, M_e, cell_model, stimulus={0: I_s})

if __name__ == "__main__":
    main()
