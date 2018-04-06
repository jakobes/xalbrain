from dolfin import *

mesh = UnitSquareMesh(10, 10)
element = FiniteElement("CG", mesh.ufl_cell(), 1)

V = FunctionSpace(mesh, element)
W = FunctionSpace(mesh, MixedElement((element, element)))

# Test and trial functions
v, ue = TrialFunctions(W)
w, q = TestFunctions(W)

vue = Function(W)
# Solution on previous timestep
merger = FunctionAssigner(V, W.sub(0))
v_ = Function(V)

# Conductivity Tensors
Me = Constant(1)
Mi = Constant(1)

dt = 0.001
T = 0.1
time = 0.0
dtc = Constant(dt)

ac_str = "cos(t)*cos(2*pi*x[0])*cos(2*pi*x[1]) + 4*pow(pi, 2)*cos(2*pi*x[0])*cos(2*pi*x[1])*sin(t)"
stimulus = Expression(ac_str, t=time, degree=3)

Dt_v = (v - v_)/dtc

G = Dt_v*w*dx
G += inner(Mi*grad(v), grad(w))*dx + inner(Mi*grad(ue), grad(w))*dx
G += inner(Mi*grad(v), grad(q))*dx + inner((Mi + Me)*grad(ue), grad(q))*dx
G -= stimulus*w*dx

a, L = system(G)
A = assemble(a)

solver = PETScKrylovSolver("gmres", "petsc_amg")
solver.set_operator(A)

while time <= T:
    time += dt
    stimulus.t = time 
    b = assemble(L)
    b -= b.sum()/b.size()
    solver.solve(vue.vector(), b)
    merger.assign(v_, vue.sub(0))

    V, UE = vue.split(deepcopy=True)
    print(UE.vector().norm("l2"), V.vector().norm("l2"))
