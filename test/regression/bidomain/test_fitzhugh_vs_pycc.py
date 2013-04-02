"""
This test tests the splitting solver for the bidomain equations with a
FitzHughNagumo model.

The test case was been compared against pycc up till T = 100.0. The
relative difference in L^2(mesh) norm between beat and pycc was then
less than 0.2% for all timesteps in all variables.

The test was then shortened to T = 4.0, and the reference at that time
computed and used as a reference here.
"""

__author__ = "Marie E. Rognes (meg@simula.no), 2012--2013"
__all__ = []

import math

from dolfin import *
from beatadjoint import *

parameters["reorder_dofs_serial"] = False
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True

class InitialCondition(Expression):
    def eval(self, values, x):
        r = math.sqrt(x[0]**2 + x[1]**2)
        values[1] = 0.0
        if r < 0.25:
            values[0] = 30.0
        else:
            values[0] = -85.0
    def value_shape(self):
        return (2,)

class MyHeart(CardiacModel):
    def __init__(self, cell_model):
        CardiacModel.__init__(self, cell_model)
    def domain(self):
        return UnitSquareMesh(100, 100)
    def conductivities(self):
        chi = 2000.0   # cm^{-1}
        s_il = 3.0/chi # mS
        s_it = 0.3/chi # mS
        s_el = 2.0/chi # mS
        s_et = 1.3/chi # mS
        M_i = as_tensor(((s_il, 0), (0, s_it)))
        M_e = as_tensor(((s_el, 0), (0, s_et)))
        return (M_i, M_e)

def setup_model():
    "Set-up cardiac model based on a slightly non-standard set of parameters."

    k = 0.00004; Vrest = -85.; Vthreshold = -70.; Vpeak = 40.;
    v_amp = Vpeak - Vrest
    l = 0.63; b = 0.013;
    cell_parameters = {"c_1": k*v_amp**2, "c_2": k*v_amp, "c_3": b/l,
                       "a": (Vthreshold - Vrest)/v_amp, "b": l,
                       "v_rest":Vrest, "v_peak": Vpeak}

    cell = FitzHughNagumoManual(cell_parameters)
    heart = MyHeart(cell)

    return heart

if __name__ == "__main__":

    # Set-up cardiac model
    heart = setup_model()

    # Set-up solver
    ps = SplittingSolver.default_parameters()
    ps["enable_adjoint"] = True
    ps["linear_variational_solver"]["linear_solver"] = "direct"
    ps["theta"] = 1.0
    ps["ode_theta"] = 0.5
    ps["ode_polynomial_family"] = "CG"
    ps["ode_polynomial_degree"] = 1
    solver = SplittingSolver(heart, ps)

    # Define end-time and (constant) timestep
    dt = 0.25 # mS
    T = 4.0 + 1.e-6  # mS

    # Define initial condition(s)
    ic = InitialCondition()
    vs0 = project(ic, solver.VS)
    (vs_, vs, u) = solver.solution_fields()
    vs_.assign(vs0)

    # Solve
    info_green("Solving primal")
    total = Timer("XXX: Total solver time")
    solutions = solver.solve((0, T), dt)
    for (timestep, vs, u) in solutions:
        plot(u)
        continue
    total.stop()
    list_timings()

    norm_u = norm(u)
    print "norm_u = %.16f" % norm_u
    reference =  11.2482303059560245
    diff = abs(reference - norm_u)
    tol = 1.e-12
    assert (diff < tol), "Mismatch: %r" % diff
