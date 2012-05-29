from dolfin import *
from dolfin_adjoint import *

# Initial set-up
mesh = UnitSquare(2, 2)
VS = VectorFunctionSpace(mesh, "CG", 1, 2)
VU = VectorFunctionSpace(mesh, "CG", 1, 2)

# Set-up initial condition
vs_expr = Expression(("1.0", "0.0"))
vs0 = project(vs_expr, VS, annotate=True)

# Function at previous time
vs_ = Function(VS)
vs_.assign(vs0, annotate=True)
(v_, s_) = split(vs_)

# Function at this time
vs = Function(VS)
vs.assign(vs_, annotate=True) # Start with a good guess for Newton
(v, s) = split(vs)

# Set-up and solve system 1
(w, r) = TestFunctions(VS)
Dt_v = (v - v_);
Dt_s = (s - s_)
G = (Dt_v + (v- s))*w*dx + inner(Dt_s - (v - s), r)*dx
solve(G == 0, vs, annotate=True)

# Set-up system 2
v_hat = vs[0]
(v, u) = TrialFunctions(VU)
(w, q) = TestFunctions(VU)
Dt_v = (v - v_hat)
G = Dt_v*w*dx + inner(grad(v), grad(q))*dx + inner(grad(u), grad(q))*dx
a, L = system(G)

# Solve system 2
vu = Function(VU)
solve(a == L, vu, bcs=None, annotate=True)

plot(vu[0])
plot(vu[1])
interactive()

info_green("Replaying")
success = replay_dolfin(tol=1.e-10, stop=True)#, forget=False)
