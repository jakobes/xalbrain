"""
"""

# Marie E. Rognes <meg@simula.no>
# Last changed: 2012-10-11

import math
from dolfin import *
from dolfin_adjoint import *
from beatadjoint import *

from fitzhugh_demo import InitialCondition, MyHeart, cell_model

parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True

# Setup cell and cardiac model
cell = cell_model()
heart = MyHeart(cell)

# Set-up solver
ps = SplittingSolver.default_parameters()
ps["enable_adjoint"] = True
ps["linear_variational_solver"]["linear_solver"] = "direct"
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

