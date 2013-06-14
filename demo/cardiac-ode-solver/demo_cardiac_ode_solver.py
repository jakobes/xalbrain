"""
FIXME
"""

__author__ = "Marie E. Rognes (meg@simula.no), 2013"

from beatadjoint import *

# For computing faster
parameters["form_compiler"]["cpp_optimize"] = True
flags = "-O3 -ffast-math -march=native"
parameters["form_compiler"]["cpp_optimize_flags"] = flags

def forward():
    info_green("Running forward model")

    # Set-up domain in space and time
    N = 100
    mesh = UnitSquareMesh(N, N)
    time = Constant(0.0)

    # Choose your favorite cell model and extract some info
    model = FitzHughNagumoManual()
    num_states = model.num_states()
    F = model.F
    I = model.I

    # Add some forces
    stimulus = Expression("t", t=time)

    # Choose your favorite solver
    params = CardiacODESolver.default_parameters()
    params["scheme"] = "RK4"
    solver = CardiacODESolver(mesh, time, num_states, F, I,
                              I_s=stimulus, params=params)

    # Set-up initial conditions
    (vs_, vs) = solver.solution_fields()
    # vs_.assign(model.initial_conditions())

    # Set-up other solution parameters
    dt = 0.2
    interval = (0.0, 1.0)

    # Generator for solutions
    solutions = solver.solve(interval, dt)

    # Do something with the solutions
    times = []
    values = []
    for ((t0, t1), vs) in solutions:
        times.append(t1)
        print vs.vector().array()

def replay():
    info_green("Replaying forward model")

    # Output some html
    adj_html("forward.html", "forward")

    # Replay
    parameters["adjoint"]["stop_annotating"] = True
    success = replay_dolfin(tol=0.0, stop=True)
    if success:
        info_green("Replay successful")
    else:
        info_green("Replay failed")

if __name__ == "__main__":

    # Run forward model
    forward()

    # Replay
    replay()
