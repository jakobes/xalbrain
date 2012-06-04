from dolfin import *
from dolfin_adjoint import *

# Cardiac solver specific imports
from splittingsolver import *
from models import *

class MyHeart(CardiacModel):
    def __init__(self, cell_model):
        CardiacModel.__init__(self, cell_model)

    def domain(self):
        return Mesh("mesh_1.xml.gz")

    def conductivities(self):
        M_i = 1.0
        M_e = 1.0
        return (M_i, M_e)

cell = FitzHughNagumo()
heart = MyHeart(cell)

application_parameters = Parameters()
application_parameters.add("theta", 0.5)
application_parameters.add("enable_adjoint", True)

solver = SplittingSolver(heart, application_parameters)

def main(vs0):
    (vs_, vs, u) = solver.solution_fields()
    vs_.assign(vs0, annotate=application_parameters["enable_adjoint"])
    solver.solve((0, 0.1), 0.01)
    return (vs, vs_)

# Define initial conditions
vs_expr = Expression(("0.0", "0.0"))
vs0 = project(vs_expr, solver.VS)

# Run main stuff
info_green("Solving primal")
(vs, vs_) = main(vs0)

# Try replaying forward
replay = True
if replay:
    info_green("Replaying")
    success = replay_dolfin(tol=1.e-10, stop=True, forget=False)

# # Try computing some kind of adjoint
# j = inner(vs, vs)*dx
# J = FinalFunctional(j)
# parameters["adjoint"]["stop_annotating"] = True # stop registering equations

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

