"""
"""

# Marie E. Rognes <meg@simula.no>
# Last changed: 2012-10-26

import math

from dolfin import *
from beatadjoint import *

parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True

class Cond(Expression):
    def eval(self, values, x):
        origin = (0.2, 0.5)
        r = 0.1**2
        radius = ((x[0] - origin[0])**2 + (x[1] - origin[1])**2)
        if radius < r:
            values[0] = self.value/self.factor
        else:
            values[0] = self.value

N = 50
mesh = UnitSquare(N, N)

class MyHeart(CardiacModel):
    def __init__(self, cell_model):
        CardiacModel.__init__(self, cell_model)
    def domain(self):
        return mesh
    def conductivities(self):
        chi = 1400.0   # cm^{-1}
        s_il = Cond()
        s_il.value = 1.74/chi
        s_il.factor = 15.0
        s_it = Cond()
        s_it.value = 0.192/chi # mS
        s_it.factor = 2.0
        s_el = Cond()
        s_el.value = 3.125/chi
        s_el.factor = 2.0
        s_et = Cond()
        s_et.value = 1.18/chi # mS
        s_et.factor = 2.0

        M_i = as_tensor(((s_il, 0), (0, s_it)))
        M_e = as_tensor(((s_el, 0), (0, s_et)))
        return (M_i, M_e)

def cell_model():
    # Set-up cell model
    k = 0.00004;
    Vrest = -85.;
    Vthreshold = -70.;
    Vpeak = 40.;
    k = 0.00004;
    l = 0.63;
    b = 0.013;
    v_amp = Vpeak - Vrest
    cell_parameters = {"c_1": k*v_amp**2, "c_2": k*v_amp, "c_3": b/l,
                       "a": (Vthreshold - Vrest)/v_amp, "b": l,
                       "v_rest":Vrest, "v_peak": Vpeak}
    model = OriginalFitzHughNagumo(cell_parameters)
    return model

class Stimulus(Expression):
    def eval(self, values, x):
        if (x[0] <= 0.01 and self.t <= 10.0):
            values[0] = 70.0
        else:
            values[0] = 0.0

if __name__ == "__main__":

    cell = OriginalFitzHughNagumo()#cell_model()
    #cell = cell_model()

    # Set-up cardiac model
    heart = MyHeart(cell)
    pulse = Stimulus()
    heart.stimulus = pulse

    # Set-up solver: direct solver is way quicker for this case with
    # constant time step and small size
    ps = SplittingSolver.default_parameters()
    ps["enable_adjoint"] = True
    ps["linear_variational_solver"]["linear_solver"] = "direct"
    solver = SplittingSolver(heart, ps)

    # Define end-time and (constant) timestep
    dt = 1.0 # mS
    T = 100.0 + 1.e-6  # mS

    # Define initial condition(s)
    ic = cell.initial_conditions()
    vs0 = project(ic, solver.VS)
    (vs_, vs, u) = solver.solution_fields()
    vs_.assign(vs0)

    # Storage
    v_pvd = File("results/v.pvd")
    u_pvd = File("results/u.pvd")
    s_pvd = File("results/s.pvd")

    # Solve
    info_green("Solving primal")
    solutions = solver.solve((0, T), dt)
    for (timestep, vs, u) in solutions:
        (v, s) = vs.split(deepcopy=True)

        print v.vector().max()
        v_pvd << v
        u_pvd << u
        s_pvd << s

    plot(v)
    interactive()
