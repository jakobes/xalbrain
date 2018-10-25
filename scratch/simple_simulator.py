from dolfin import *

dt = 0.1
t = 0
T = 1

mesh = UnitCubeMesh(10, 10, 10)

F_ele = FiniteElement("CG", mesh.ufl_cell(), 1)
W = FunctionSpace(mesh, MixedElement((F_ele, F_ele)))

u, v = TrialFunctions(W)
w, q = TestFunctions(W)

Mi = Constant(1)
Me = Constant(1)

current_UV = Function(W)        # This time step
prev_UV = Function(W)       # previous time step

current_UV.vector().array()[:] = 0

prev_U, prev_V = split(prev_UV)

dtc = Constant(dt)

G = (v - prev_V)/dtc*w*dx
G += inner(Mi*grad(v), grad(w))*dx
G += inner(Mi*grad(u), grad(w))*dx

G += inner(Mi*grad(v), grad(q))*dx
G += inner((Mi + Me)*grad(u), grad(q))*dx

lhs, rhs = system(G)
solver = PETScKrylovSolver("gmres")
solver.set_operator(assemble(lhs))

# Parameters
solver.parameters.absolute_tolerance = None
solver.parameters.convergence_norm_type = None
solver.parameters.divergence_limit = None
solver.parameters.error_on_nonconvergence = None
solver.parameters.maximum_iterations = None
solver.parameters.monitor_convergence = True
solver.parameters.relative_tolerance = None

solver.parameters.nonzero_initial_guess = True
solver.parameters.report = True


assign_space = FunctionSpace(mesh, "CG", 1)
assigning_func = project(Constant(-70), assign_space)
assigner = FunctionAssigner(W.sub(1), assign_space)
assigner.assign(prev_UV.sub(1), assigning_func)

assigner = FunctionAssigner(W.sub(1), assign_space)
assigner.assign(current_UV.sub(1), assigning_func)


while t < T + 1e-3:
    b = assemble(rhs)
    solver.solve(current_UV.vector(), b)
    print(current_UV.vector().array())
    t += dt
