"""
"""

# Marie E. Rognes <meg@simula.no>
# Last changed: 2012-10-26

import sys
import math
from dolfin import *
from dolfin_adjoint import *
import time

assert(len(sys.argv) > 1), "Please give N (int), mesh size as command-line arg"

set_log_level(DEBUG)

# Adjust some general parameters
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True

# Some parameters
a = 0.13; b = 0.013; c_1 = 0.26; c_2 = 0.1; c_3 = 1.0;
v_rest = -85.; v_peak = 40.0
v_amp = v_peak - v_rest
v_th = v_rest + a*v_amp

# Domain
N = int(sys.argv[1])
domain = UnitCube(N, N, N)

# Construct conductivity tensors
chi = 1400
V = FunctionSpace(domain, "CG", 1)
g = Function(V)
g.vector()[:] = 2.0/chi
M_e = diag(as_vector([g, g, g]))
M_i = diag(as_vector([g, g, g]))

# class MyHeart(CardiacModel):
#     def __init__(self, cell_model):
#         CardiacModel.__init__(self, cell_model)
#     def domain(self):
#         return domain
#     def conductivities(self):
#         return (M_i, M_e)

# # Setup cell and cardiac model
# cell = OriginalFitzHughNagumo()
# heart = MyHeart(cell)

# Define some simulation protocol (use cpp expression for speed)
pulse = Expression("amp*x[0]*(1-x[0])*x[1]*(1-x[1])*x[2]*t", t=0.0, amp=100.0)

# Set-up solver
# -----------------------------------------------------------------------------

# Set-up function spaces
V = FunctionSpace(domain, "CG", 1)
R = FunctionSpace(domain, "R", 0)
VUR = MixedFunctionSpace([V, V, R])
S = FunctionSpace(domain, "DG", 0)
VS = V*S

theta = 0.5

def ode_step(interval, ics):
    # Extract time domain
    (t0, t1) = interval
    k_n = Constant(t1 - t0)

    # Extract initial conditions
    (v_, s_) = split(ics)

    # Set-up current variables
    vs = Function(VS, name="ode_vs")
    vs.assign(ics, annotate=True) # Start with good guess
    (v, s) = split(vs)
    (w, r) = TestFunctions(VS)

    # Define equation based on cell model
    Dt_v = (v - v_)/k_n
    Dt_s = (s - s_)/k_n

    F = lambda v, s: b*(-c_3*s + v - v_rest)
    I_ion = lambda v, s: (c_1/(v_amp**2)*(v - v_rest)*(v - v_th)*(v_peak - v)
                          - c_2/(v_amp)*(v - v_rest)*s)
    I_theta = - (theta*I_ion(v, s) + (1 - theta)*I_ion(v_, s_))
    F_theta = theta*F(v, s) + (1 - theta)*F(v_, s_)

    t = t0 + theta*(t1 - t0)
    pulse.t = t
    I_theta += pulse

    # Set-up system
    G = (Dt_v - I_theta)*w*dx + inner(Dt_s - F_theta, r)*dx
    pde = NonlinearVariationalProblem(G, vs, J=derivative(G, vs))

    # Set-up solver
    solver = NonlinearVariationalSolver(pde)
    solver.parameters["linear_solver"] = "iterative"

    # Solve system
    solver.solve(annotate=True)
    return vs

def pde_variational_form(k_n, vs_):

    # Define variational formulation
    (v, u, l) = TrialFunctions(VUR)
    (w, q, lamda) = TestFunctions(VUR)

    # Set-up variational problem
    (v_, s_) = split(vs_)
    Dt_v = (v - v_)
    theta_parabolic = (theta*inner(M_i*grad(v), grad(w))*dx
                       + (1.0 - theta)*inner(M_i*grad(v_), grad(w))*dx
                       + inner(M_i*grad(u), grad(w))*dx)
    theta_elliptic = (theta*inner(M_i*grad(v), grad(q))*dx
                      + (1.0 - theta)*inner(M_i*grad(v_), grad(q))*dx
                      + inner((M_i + M_e)*grad(u), grad(q))*dx)

    G = (Dt_v*w*dx + k_n*theta_parabolic + theta_elliptic
         + (lamda*u + l*q)*dx)

    (a, L) = system(G)
    return (a, L)


# Helper functions
vs_ = Function(VS, name="vs_")
vs = Function(VS, name="vs")
vur = Function(VUR)

# Set-up initial conditions
ic = Expression(("V", "S"), V=-85.0, S=0.0)
vs_.assign(ic, annotate=False)

timestep = 1.0
t0 = 0.0
t1 = timestep
T = 2*timestep

(a, L) = pde_variational_form(timestep, vs_)
A = assemble(a, annotate=True)

pde_solver = LUSolver(A)
pde_solver.parameters["reuse_factorization"] = True
pde_solver.parameters["same_nonzero_pattern"] = True

info_green("Solving forward")
while (t1 <= T + DOLFIN_EPS):

    t = t0 + theta*timestep

    # Solve tentative ode-step:
    info_blue("Tentative ODE")
    vs_star = ode_step((t0, t), vs_)

    # Solve pde step
    info_blue("PDE step")
    vs_.assign(vs_star, annotate=True)
    rhs = assemble(L, annotate=True)
    pde_solver.solve(vur.vector(), rhs, annotate=True)

    # Merge step
    info_blue("Merging")
    (v, u, r) = split(vur)
    (v_star, s_star) = split(vs_star)
    v_s_star = project(dolfin.as_vector((v, s_star)), VS, annotate=True,
                       solver_type="cg")

    # Solve corrective ode-step
    info_blue("Corrective ODE")
    vs = ode_step((t, t1), v_s_star)

    # Update
    info_blue("Updating")
    vs_.assign(vs, annotate=True)
    t0 = t1
    t1 = t1 + timestep

#set_log_level(DEBUG)

info_green("Computing gradient")
J = Functional(inner(vs, vs)*dx*dt[FINISH_TIME])
dJdg = compute_gradient(J, InitialConditionParameter(g))


