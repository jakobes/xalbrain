from dolfin import *
from dolfin_adjoint import *

# Initial set-up
N = 16
mesh = UnitSquare(N, N)
V = VectorFunctionSpace(mesh, "CG", 1, 2)

# Set-up initial condition
vs_expr = Expression(("1.0", "0.0"))
vs0 = project(vs_expr, V)

# Function at previous time
vs_ = Function(V)
vs_.assign(vs0)
(v_, s_) = split(vs_)

# Function at this time
vs = Function(V)
vs.assign(vs_) # Uncomment this and replay is exact
(v, s) = split(vs)

# Set-up and solve system 1
(w, r) = TestFunctions(V)
Dt_v = (v - v_);
Dt_s = (s - s_)
G = (Dt_v + (v - s))*w*dx + inner(Dt_s - (v - s), r)*dx
solve(G == 0, vs)

# Set-up system 2
(v, u) = TrialFunctions(V)
(w, q) = TestFunctions(V)
Dt_v = (v - vs[0])
G = Dt_v*w*dx + inner(grad(v), grad(q))*dx + inner(grad(u), grad(q))*dx
a, L = system(G)

# Solve system 2
vu = Function(V)
print "vu = ", vu
solve(a == L, vu, bcs=DirichletBC(V, (0.0, 0.0), "near(x[0], 1.0)"))

plot(vu[0])
plot(vu[1])
interactive()

info_green("Replaying")
success = replay_dolfin(tol=1.e-10, stop=True)
