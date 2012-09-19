from dolfin import *
import math

n = 2#00
mesh = UnitSquare(n, n)

class Circle(SubDomain):
    #def __init__(self, p, r):
    #    self.p = p
    #    self.r = r
    def inside(self, x, on_boundary):
        self.r = 0.1
        self.p = [0.5, 0.5]
        radius = math.sqrt((x[0] - self.p[0])**2 + (x[1] - self.p[1])**2)
        return radius < self.r

V = FunctionSpace(mesh, "CG", 1)
R = FunctionSpace(mesh, "R", 0)
Rn = VectorFunctionSpace(mesh, "R", 0)

x = V.cell().x

r_value = 0.1
p_vec = [0.5, 0.5]

# Center of bump
p = Function(Rn)
p.vector()[0] = p_vec[0]
p.vector()[1] = p_vec[1]

# Radius of bump
r = Function(R)
r.vector()[:] = r_value

circle = Circle()#p_vec, r_value)

v = TrialFunction(V)

# Experimenting with conditionals
condition = le(inner((x - p), (x - p)), r**2)
true_value = exp(- (r**2/(r**2 - inner((x - p), (x - p)))))
#true_value = p[0]
false_value = 0.0
M = conditional(condition, true_value, false_value)
#step = conditional(condition, 1.0, 0.0)
#M = true_value*step

#M = exp(- (r**2/(r**2 - inner((x - p), (x - p)))))
zero = Function(R)

markers = CellFunction("uint", mesh)
markers.set_all(0)
circle.mark(markers, 1)

dxx = Measure("dx")[markers]

#plot(markers, interactive=True)

#L = M*v*dxx(1)
L = M*v*dx

F = derivative(L, p, p)
#(a, L) = system(F)

A = assemble(F)
info(A, True)
#print "A = ", A
#print A.norm("frobenius")

#A = assemble(a)
#b = assemble(L)

#print a


