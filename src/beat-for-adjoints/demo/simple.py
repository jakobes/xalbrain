from dolfin import *
from dolfin_adjoint import *

# Cardiac solver specific imports
from splittingsolver import *
from models import *

class MyHeart(CardiacModel):
    def __init__(self, cell_model):
        CardiacModel.__init__(self, cell_model)

    def domain(self):
        N = 16
        return UnitSquare(N, N)

    def conductivities(self):
        M_i = 1.0
        M_e = 1.0
        return (M_i, M_e)

cell_parameters = {"epsilon": 0.01, "gamma": 0.5, "alpha": 0.1}
cell = FitzHughNagumo(cell_parameters)
heart = MyHeart(cell)

parameters = Parameters()
parameters.add("theta", 0.5)
parameters.add("enable_adjoint", True)
solver = SplittingSolver(heart, parameters)

# Set initial conditions
vs_expr = Expression(("- x[0]*(1-x[0])*x[1]*(1-x[1])", "0.0"))
vs0 = project(vs_expr, solver.VS,
              annotate=parameters["enable_adjoint"])
(vs_, vs, u) = solver.solution_fields()
vs_.assign(vs0, annotate=parameters["enable_adjoint"])

# Just quick regression test, not validation
info_green("Solving primal")
solver.solve((0, 0.1), 0.01)
print "-"*80
ref = a = 0.023771346883295
a = norm(vs.split()[0])
print "a = %.15f" % a
diff = abs(a - ref)
assert diff < 1.e-9, "a = %g, diff = %g" % (a, diff)
print "-"*80

adj_html("forward.html", "forward")
adj_html("adjoint.html", "adjoint")
info_green("Replaying")
success = replay_dolfin(tol=1.e-10, stop=True)#, forget=False)

J = FinalFunctional(inner(vs, vs)*dx)

