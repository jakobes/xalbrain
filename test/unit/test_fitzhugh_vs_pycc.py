"""
This test tests the splitting solver for the bidomain equations with a
FitzHughNagumo model.

The test case was been compared against pycc up till T = 100.0. The
relative difference in L^2(mesh) norm between beat and pycc was then
less than 0.2% for all timesteps in all variables.

The test was then shortened to T = 4.0, and the reference at that time
computed and used as a reference here.
"""

__author__ = "Marie E. Rognes (meg@simula.no), 2012--2014"
__all__ = []

import math

from beatadjoint import Expression, parameters, FitzHughNagumoManual, as_tensor
from beatadjoint import UnitSquareMesh, Constant, CardiacModel
from beatadjoint import BasicSplittingSolver, project, norm

from testutils import assert_almost_equal, slow

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

def setup_model():
    "Set-up cardiac model based on a slightly non-standard set of parameters."

    # Define cell parameters
    k = 0.00004; Vrest = -85.; Vthreshold = -70.; Vpeak = 40.;
    v_amp = Vpeak - Vrest
    l = 0.63; b = 0.013;
    cell_parameters = {"c_1": k*v_amp**2, "c_2": k*v_amp, "c_3": b/l,
                       "a": (Vthreshold - Vrest)/v_amp, "b": l,
                       "v_rest":Vrest, "v_peak": Vpeak}
    cell = FitzHughNagumoManual(cell_parameters)

    # Define conductivities
    chi = 2000.0   # cm^{-1}
    s_il = 3.0/chi # mS
    s_it = 0.3/chi # mS
    s_el = 2.0/chi # mS
    s_et = 1.3/chi # mS
    M_i = as_tensor(((s_il, 0), (0, s_it)))
    M_e = as_tensor(((s_el, 0), (0, s_et)))

    # Define mesh
    domain = UnitSquareMesh(100, 100)
    time = Constant(0.0)

    heart = CardiacModel(domain, time, M_i, M_e, cell)
    return heart

@slow
def test_fitzhugh():

    # Set-up cardiac model
    heart = setup_model()

    # Set-up solver
    ps = BasicSplittingSolver.default_parameters()
    ps["BasicBidomainSolver"]["linear_variational_solver"]["linear_solver"] = "direct"
    ps["BasicBidomainSolver"]["theta"] = 1.0
    ps["theta"] = 1.0
    ps["BasicCardiacODESolver"]["S_polynomial_family"] = "CG"
    ps["BasicCardiacODESolver"]["S_polynomial_degree"] = 1
    ps["BasicCardiacODESolver"]["V_polynomial_family"] = "CG"
    ps["BasicCardiacODESolver"]["V_polynomial_degree"] = 1
    solver = BasicSplittingSolver(heart, ps)

    # Define end-time and (constant) timestep
    dt = 0.25 # mS
    T = 4.0 + 1.e-6  # mS

    # Define initial condition(s)
    ic = InitialCondition()
    vs0 = project(ic, solver.VS)
    (vs_, vs, u) = solver.solution_fields()
    vs_.assign(vs0)

    # Solve
    solutions = solver.solve((0, T), dt)
    for (timestep, (vs_, vs, vur)) in solutions:
        continue

    u = project(vur[1], vur.function_space().sub(1).collapse())
    norm_u = norm(u)
    reference =  11.2487499749304991
    print "norm_u = ", norm_u
    print "reference = ", reference

    assert_almost_equal(reference, norm_u, 1.e-6)
