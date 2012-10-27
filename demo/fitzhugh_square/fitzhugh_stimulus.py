"""
"""

# Marie E. Rognes <meg@simula.no>
# Last changed: 2012-10-27

import math

from dolfin import *
from beatadjoint import *

parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True

class Cond(Expression):
    def eval(self, values, x):
        origin = (0.2, 0.5)
        r = 0.1**2
        radius = ((x[0] - origin[0])**2 + (x[1] - origin[1])**2)
        if radius < r:
            values[0] = self.value/self.factor
        else:
            values[0] = self.value

N = 50
mesh = UnitSquare(N, N)

V = FunctionSpace(mesh, "CG", 1)

chi = 1400.0   # cm^{-1}
s_il = Cond()
s_il.value = 1.74/chi
s_il.factor = 15.0
s_il_var = Function(project(s_il, V, annotate=False), name="s_il_var")

s_it = Cond()
s_it.value = 0.192/chi # mS
s_it.factor = 2.0
s_it_var = Function(project(s_it, V, annotate=False), name="s_it_var")

s_el = Cond()
s_el.value = 3.125/chi
s_el.factor = 2.0
s_el_var = Function(project(s_el, V, annotate=False), name="s_el_var")

s_et = Cond()
s_et.value = 1.18/chi # mS
s_et.factor = 2.0
s_et_var = Function(project(s_et, V, annotate=False), name="s_et_var")

# Darker magical trick recommended by Patrick
s_il = Function(V, name="s_il")
s_il.assign(s_il_var, annotate=True, force=True)
s_it = Function(V, name="s_it")
s_it.assign(s_it_var, annotate=True, force=True)

s_et = Function(V, name="s_et")
s_et.assign(s_et_var, annotate=True, force=True)
s_el = Function(V, name="s_el")
s_el.assign(s_el_var, annotate=True, force=True)

class MyHeart(CardiacModel):
    def __init__(self, cell_model):
        CardiacModel.__init__(self, cell_model)
    def domain(self):
        return mesh
    def conductivities(self):
        M_i = as_tensor(((s_il, 0), (0, s_it)))
        M_e = as_tensor(((s_el, 0), (0, s_et)))
        return (M_i, M_e)

def cell_model():
    # Set-up cell model
    k = 0.00004;
    Vrest = -85.;
    Vthreshold = -70.;
    Vpeak = 40.;
    k = 0.00004;
    l = 0.63;
    b = 0.013;
    v_amp = Vpeak - Vrest
    cell_parameters = {"c_1": k*v_amp**2, "c_2": k*v_amp, "c_3": b/l,
                       "a": (Vthreshold - Vrest)/v_amp, "b": l,
                       "v_rest":Vrest, "v_peak": Vpeak}
    model = OriginalFitzHughNagumo(cell_parameters)
    return model

class Stimulus(Expression):
    def eval(self, values, x):
        if (x[0] <= 0.01 and self.t <= 10.0):
            values[0] = 30.0
        else:
            values[0] = 0.0

if __name__ == "__main__":


    cell = OriginalFitzHughNagumo()#cell_model()
    #cell = cell_model()

    # Set-up cardiac model
    heart = MyHeart(cell)
    pulse = Stimulus()
    heart.stimulus = pulse

    # Set-up solver: direct solver is way quicker for this case with
    # constant time step and small size
    ps = SplittingSolver.default_parameters()
    ps["enable_adjoint"] = True
    ps["linear_variational_solver"]["linear_solver"] = "direct"
    solver = SplittingSolver(heart, ps)

    # Define end-time and (constant) timestep
    k_n = 1.0 # mS
    T = 40.0 + 1.e-6  # mS

    # Define initial condition(s)
    ic = cell.initial_conditions()
    vs0 = project(ic, solver.VS)
    (vs_, vs, u) = solver.solution_fields()
    vs_.assign(vs0)

    # Storage
    v_pvd = File("results/v.pvd")
    u_pvd = File("results/u.pvd")
    s_pvd = File("results/s.pvd")

    # Solve
    info_green("Solving primal")
    solutions = solver.solve((0, T), k_n)
    for (timestep, vs, u) in solutions:
        #(v, s) = vs.split(deepcopy=True)
        #print v.vector().max()
        #v_pvd << v
        #u_pvd << u
        #s_pvd << s
        continue

    (v, s) = split(vs)
    plot(v, title="v")

    V = solver.VS.sub(0).collapse()
    v_obs = Function(V, "v40.xml.gz")
    plot(v_obs, title="v_obs")

    #(v_store, s_store) = vs.split(deepcopy=True)
    #file = File("v40.xml.gz")
    #file << v_store

    adj_html("forward.html", "forward")
    adj_html("adjoint.html", "adjoint")

    info_green("Computing gradient wrt s_el_var")
    J = Functional(inner(v - v_obs, v - v_obs)*dx*dt[FINISH_TIME])
    dJds_el = compute_gradient(J, InitialConditionParameter(s_el_var), forget=False)
    plot(dJds_el, title="Sensitivity wrt s_el")

    info_green("Computing gradient wrt s_et_var")
    dJds_et = compute_gradient(J, InitialConditionParameter(s_et_var), forget=False)
    plot(dJds_et, title="Sensitivity srt s_et")

    info_green("Computing gradient wrt s_il_var")
    dJds_il = compute_gradient(J, InitialConditionParameter(s_il_var), forget=False)

    plot(dJds_il, title="Sensitivity srt s_il")

    info_green("Computing gradient wrt s_it_var")
    dJds_it = compute_gradient(J, InitialConditionParameter(s_it_var), forget=False)
    plot(dJds_it, title="Sensitivity srt s_it")
    interactive()

    exit()
