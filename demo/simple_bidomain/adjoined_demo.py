"""
"""

# Marie E. Rognes <meg@simula.no>
# Last changed: 2012-10-03

import math
from dolfin import *
from dolfin_adjoint import *
from beatadjoint import *

#set_log_level(WARNING)

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
        n = 100
        return UnitSquare(n, n)
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
ps = SplittingSolver.default_parameters()
ps["enable_adjoint"] = True
solver = SplittingSolver(heart, parameters=ps)

# Define end-time and (constant) timestep
T = 2.0
k_n = 0.25

# Split solve out into separate function for Taylor test purposes
def main(ic):
    # Assign initial condition
    (vs_, vs, u) = solver.solution_fields()
    vs_.adj_name = "vs_"
    vs.adj_name = "vs"
    u.adj_name = "u"
    vs_.assign(ic, annotate=True)

    # Solve
    begin("Solving primal")
    solutions = solver.solve((0, T), k_n)
    for (timestep, vs, u) in solutions:
        continue
    end()

    return (vs, u)

if __name__ == "__main__":

    # Define initial condition here (no need to annotate this step)
    ic = InitialCondition()
    ic = Function(project(ic, solver.VS, annotate=False))

    # Run forward simulation once
    (vs, u) = main(ic)

    # Stop annotating here (done with forward solve)
    parameters["adjoint"]["stop_annotating"] = True

    # Compute value of functional at "end"
    J_value = assemble(inner(vs, vs)*dx)

    # Check replay
    info_green("Replaying")
    success = replay_dolfin(tol=0.0, stop=True)

    # Store record
    adj_html("forward.html", "forward")
    adj_html("adjoint.html", "adjoint")

    # Define some functional
    J = Functional(inner(vs, vs)*dx*dt[FINISH_TIME])

    # Compute gradient of J with respect to vs_
    info_green("Computing gradient")
    dJdic = compute_gradient(J, InitialConditionParameter("vs_"), forget=False)
    assert dJdic is not None

    # Run Taylor test
    info_green("Verifying")
    def Jhat(ic):
        (vs, u) = main(ic)
        return assemble(inner(vs, vs)*dx)
    minconv = taylor_test(Jhat, InitialConditionParameter("vs_"),J_value, dJdic)
    print "Minimum convergence rate: ", minconv
