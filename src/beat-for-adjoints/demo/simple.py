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
parameters.add("theta", 1.0)
parameters.add("enable_adjoint", True)
solver = SplittingSolver(heart, parameters)

# Set initial conditions
vs_expr = Expression(("- x[0]*(1-x[0])*x[1]*(1-x[1])", "0.0"))
vs_ = project(vs_expr, solver.VS,
              annotate=parameters["enable_adjoint"])
(vs, u) = solver.solution_fields()
vs.assign(vs_, annotate=parameters["enable_adjoint"])

solver.solve((0, 0.1), 0.01)

# Just quick regression test, not validation
#solver.solve((0, 0.1), 0.01)
#print "-"*80
#ref = 0.028030781489032
#a = norm(vs.split()[0])
#diff = abs(a - ref)
#assert diff < 1.e-9, "a = %g, diff = %g" % (a, diff)
#print "-"*80

# Try replaying
adj_html("forward.html", "forward")
adj_html("adjoint.html", "adjoint")
success = replay_dolfin(tol=1.e-15, stop=True)

J = FinalFunctional(inner(vs, vs)*dx)

