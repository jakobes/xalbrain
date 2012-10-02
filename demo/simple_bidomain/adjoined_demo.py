"""
"""

# Marie E. Rognes <meg@simula.no>
# Last changed: 2012-10-02

import math
from dolfin import *
from dolfin_adjoint import *
from beatadjoint import *

set_log_level(WARNING)

parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True

class InitialCondition(Expression):
    def eval(self, values, x):
        r = math.sqrt(x[0]**2 + x[1]**2)
        values[1] = 0.0
        if r < 0.25:
            values[0] = 30.0
        else:
            values[0] = -85.0
    def value_shape(self):
        return (2,)

class MyHeart(CardiacModel):
    def __init__(self, cell_model):
        CardiacModel.__init__(self, cell_model)
    def domain(self):
        return UnitSquare(100, 100)
    def conductivities(self):
        chi = 2000.0   # cm^{-1}
        s_il = 3.0/chi # mS
        s_it = 0.3/chi # mS
        s_el = 2.0/chi # mS
        s_et = 1.3/chi # mS
        M_i = as_tensor(((s_il, 0), (0, s_it)))
        M_e = as_tensor(((s_el, 0), (0, s_et)))
        return (M_i, M_e)

# Set-up parameters and cell model
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
cell = FitzHughNagumo(cell_parameters)

# Set-up cardiac model
heart = MyHeart(cell)

# Set-up solver
solver = SplittingSolver(heart)
solver.parameters["enable_adjoint"] = True

# Define end-time and (constant) timestep
k_n = 0.25 # mS
T = 1.0   # mS

def main(ic):

    # Assign initial condition
    (vs_, vs, u) = solver.solution_fields()
    vs_.assign(ic, annotate=True)

    # Solve
    info_green("Solving primal")
    solutions = solver.solve((0, T), k_n)
    for (timestep, vs, u) in solutions:
        continue

    return (vs, u)

if __name__ == "__main__":

    ic = InitialCondition()
    ic = project(ic, solver.VS, annotate=False)

    # Run stuff
    (vs, u) = main(ic)

    # 1: Compute value of functional at "end"
    J_value = assemble(inner(vs, vs)*dx)   # Value of functional (last u)
    print "J_value = ", J_value

    # Check replay
    info_green("Replaying")
    success = replay_dolfin(tol=0.0, stop=True)

    # Define the amount of functionals needed
    def M(vs):
        return inner(vs, vs)*dx*dt[FINISH_TIME]
    J = Functional(M(vs))  # Functional as dolfin_adjoint.Functional
    info_green("Computing gradient")

    dJdic = compute_gradient(J, InitialConditionParameter(ic), forget=False)

    print dJdic
    exit()

    def Jhat(ic):           # Functional value as function of ic
        (vs, u) = main(ic)
        return assemble(M(vs))

    parameters["adjoint"]["stop_annotating"] = True

    # Look at some Taylor test
    minconv = taylor_test(Jhat, InitialConditionParameter(ic),
                          J_value, dJdic)


