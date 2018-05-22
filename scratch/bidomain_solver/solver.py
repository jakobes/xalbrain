from dolfin import *

mesh = UnitSquareMesh(10, 10)
element = FiniteElement("CG", mesh.ufl_cell(), 1)
V = FunctionSpace(mesh, element)
W = FunctionSpace(mesh, MixedElement((element, element)))

# Test and trial functions
ue, v = TrialFunctions(W)
k, l = TestFunctions(W)

# Conductivity Tensors
Me = Constant(1)
Mi = Constant(1)

dt = 0.01
T = 0.1
time = 0.0

theta = 1.0

ac_str = "cos(t)*cos(2*pi*x[0])*cos(2*pi*x[1]) + 4*pow(pi, 2)*cos(2*pi*x[0])*cos(2*pi*x[1])*sin(t)"
stimulus = Expression(ac_str, t=time, degree=3)

stimulus_function = project(stimulus, V)
zero_function = project(Constant(0), V)

alpha = Constant(10) 
dtc =  Constant(dt)

a = ue*k*dx + dtc*inner(Me*grad(ue), grad(k))*dx \
            + dtc*inner(Me*grad(v), grad(k))*dx \
            + dtc*inner(Me*grad(ue), grad(l))*dx \
            + v*l*dx \
            + dtc*inner(Mi*grad(v), grad(l))*dx

#p = ue*k*dx + dtc*inner(Me*grad(ue), grad(k))*dx \
#    + v*l*dx + dtc*inner(Mi*grad(v), grad(l))*dx

A = assemble(a)
#P = assemble(p)


UEV = Function(W)   # soluiton on current timestep  
UEV_ = Function(W)  # solution on previous timestep  

# assigner = FunctionAssigner(W, [V, V])
# assigner.assign(UEV_, [stimulus_function, zero_function])

solver = PETScKrylovSolver("gmres", "petsc_amg")
#solver.set_operators(A, P)
solver.set_operator(A)

while time <= T + 1e-10:
    time += dt
    # UE_, V_ = split(UEV_)  
    UE_, V_ = UEV_.split()
    stimulus.t = time 
    L = UE_*k*dx + dtc*stimulus*k*dx# + dt*alpha*UE_*k*dx
    b = assemble(L)
    b -= b.sum()/b.size()
    solver.solve(UEV.vector(), b)
    UEV_.assign(UEV)
    UE, V = UEV.split(deepcopy=True)
    print(UE.vector().norm("l2"), V.vector().norm("l2"))
