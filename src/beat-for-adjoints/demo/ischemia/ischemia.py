from dolfin import *
from dolfin_adjoint import *

# Cardiac solver specific imports
from splittingsolver import *
from models import *

class MyHeart(CardiacModel):
    def __init__(self, cell_model):
        CardiacModel.__init__(self, cell_model)
        self._domain = Mesh("mesh_1.xml.gz")
        R = FunctionSpace(self._domain, "R", 0)
        self._M_i = Function(R)
        self._M_i.vector()[:] = 1.0
        self._M_e = Function(R)
        self._M_e.vector()[:] = 1.0

    def domain(self):
        return self._domain

    def conductivities(self):
        return (self._M_i, self._M_e)

cell = FitzHughNagumo()
heart = MyHeart(cell)

application_parameters = Parameters()
application_parameters.add("theta", 1.0)
application_parameters.add("enable_adjoint", True)

solver = SplittingSolver(heart, application_parameters)

def main(vs0):
    (vs_, vs, u) = solver.solution_fields()
    vs_.assign(vs0, annotate=application_parameters["enable_adjoint"])
    solver.solve((0, 0.1), 0.01)
    return (vs, vs_, u)

# Define initial conditions
vs_expr = Expression(("0.0", "0.0"))
vs0 = project(vs_expr, solver.VS, annotate=False)

# Run main stuff
info_green("Solving primal")
(vs, vs_, u) = main(vs0)

parameters["adjoint"]["stop_annotating"] = True # stop registering equations

# Try replaying forward
replay = False
if replay:
    info_green("Replaying")
    success = replay_dolfin(tol=1.e-10, stop=True, forget=False)

adj_html("forward.html", "forward")
adj_html("adjoint.html", "adjoint")

plot_solutions = False
if plot_solutions:
    plot(vs[0], title="v")
    plot(vs[1], title="s")
    plot(u, title="u")
    interactive()

# Define functional
j = inner(vs, vs)*ds
J = FinalFunctional(j)
print assemble(j)

(M_i, M_e) = heart.conductivities()
ic_param = InitialConditionParameter(M_i)
dJdM_i = compute_gradient(J, ic_param, forget=False)
plot(dJdM_i, interactive=True)
# ic_param = InitialConditionParameter(vs_) # This "works"
# #ic_param = InitialConditionParameter(vs0) # This gives None
# dJdic = compute_gradient(J, ic_param, forget=False)
# plot(dJdic, interactive=True)

# def Jhat(vs0):
#     (vs, vs_) = main(vs0)
#     j = inner(vs, vs)*dx
#     return assemble(j)

# Jvalue = assemble(j)
# print "Jvalue = ", Jvalue
# #
# conv_rate = taylor_test(Jhat, ic_param, Jvalue, dJdic)

#interactive()
