"""
Demo for propagation of electric potential through left and right
ventricles.
"""

__author__ = "Marie E. Rognes (meg@simula.no), 2012--2013"

import math
from beatadjoint import *
from demo import setup_application_parameters, setup_general_parameters
from demo import setup_cardiac_model
from demo import main as forward

import time

def main(replay=False):

    (gs, solver) = forward(store_solutions=False)

    adj_html("forward.html", "forward")
    adj_html("adjoint.html", "adjoint")

    # Stop annotating
    parameters["adjoint"]["stop_annotating"] = True

    if replay:
        info_green("Replaying")
        success = replay_dolfin(stop=True, tol=0.0)
        assert(success == True), "Non-successful replay: check the model."
        return

    # Define functional
    info_green("Using ill-versus-healthy functional")

    # Extract synthetic observed data:
    vs_obs = Function(solver.VS, name="vs_obs")
    File("forward-healthy/vs_200.xml.gz") >> vs_obs

    (vs_, vs, vu) = solver.solution_fields()
    v_obs = split(vs_obs)[0]
    v = split(vs)[0]
    J = Functional(inner(v - v_obs, v - v_obs)*dx*dt[FINISH_TIME])

    ics = [InitialConditionParameter(g) for g in gs]

    # Compute the gradient
    info_blue("Computing gradient")
    start = time.time()
    dJdg_s = compute_gradient(J, ics, forget=False)
    stop = time.time()

    # Store the results
    for (i, dJdg) in enumerate(dJdg_s):
        name = ics[i].var.name
        # plot(dJdg, title="%s" % name)
        file = File("%s/%s_sensitivity.xml.gz" % ("ill-vs-healthy-results",
                                                  name))
        file << dJdg

    # Output some timings
    print "Time for computing gradient: %g" % (stop - start)

if __name__ == "__main__":
    main()
