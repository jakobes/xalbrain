"""
This test solves the bidomain equations (assuming no state variables)
with an analytic solution to verify the correctness of the basic
splitting solver.
"""

__author__ = "Marie E. Rognes (meg@simula.no), 2012--2013"
__all__ = []


import pytest

from xalbrain import (
    CardiacModel,
    NoCellModel,
    BasicSplittingSolver,
)

from dolfin import (
    Constant,
    UnitSquareMesh,
    Function,
    Expression,
    errornorm,
)

from dolfin import __version__ as dolfin_version

from xalbrain.utils import convergence_rate

from testutils import slow

from typing import Tuple


def main(N: int, dt: float, T: float, theta: float) -> \
        Tuple[float, float, float, float, float]:
    # Create cardiac model
    mesh = UnitSquareMesh(N, N)
    time = Constant(0.0)
    cell_model = NoCellModel()

    ac_str = "cos(t)*cos(2*pi*x[0])*cos(2*pi*x[1]) + 4*pow(pi, 2)*cos(2*pi*x[0])*cos(2*pi*x[1])*sin(t)"
    stimulus = Expression(ac_str, t=time, degree=5)
    heart = CardiacModel(mesh, time, 1.0, 1.0, cell_model, stimulus=stimulus)

    # Set-up solver
    ps = BasicSplittingSolver.default_parameters()
    ps["theta"] = theta
    ps["BasicBidomainSolver"]["linear_variational_solver"]["linear_solver"] = "direct"
    solver = BasicSplittingSolver(heart, params=ps)

    # Define exact solution (Note: v is returned at end of time
    # interval(s), u is computed at somewhere in the time interval
    # depending on theta)

    v_exact =  Expression("cos(2*pi*x[0])*cos(2*pi*x[1])*sin(t)", t=T, degree=5)
    u_exact = Expression(
        "-cos(2*pi*x[0])*cos(2*pi*x[1])*sin(t)/2.0",
        t=T - (1 - theta)*dt,
        degree=5
    )

    # Define initial condition(s)
    vs0 = Function(solver.VS)
    (vs_, vs, vur) = solver.solution_fields()
    vs_.assign(vs0)

    # Solve
    solutions = solver.solve((0, T), dt)
    for (timestep, (vs_, vs, vur)) in solutions:
        continue

    # Compute errors
    v, s = vs.split(deepcopy=True)
    v_error = errornorm(v_exact, v, "L2", degree_rise=5)
    v, u, *r = vur.split(deepcopy=True)
    u_error = errornorm(u_exact, u, "L2", degree_rise=5)

    return (v_error, u_error, mesh.hmin(), dt, T)

@slow
@pytest.mark.xfail(dolfin_version == "2016.2.0", reason="Unknown")
def test_analytic_bidomain():
    "Test errors for bidomain solver against reference."

    # Create domain
    level = 0
    for level in (0, 1, 2):
        N = 10*(2**level)
        dt = 0.01/(2**level)
        T = 0.1
        v_error, u_error, h, dt, T = main(N, dt, T, 0.5)
    
        #v_reference = 4.1152719193176370e-03 # with degree = 5 and degree_rise=5
        #u_reference = 2.0271098018943513e-03 # with degree = 5 and degree_rise=5
    
        if level == 0:
            v_reference = 0.004115272130736041
            u_reference = 0.003570426889654414
        elif level == 1:
            v_reference = 0.0010651083432491597
            u_reference = 0.0030640902609339947
        elif level == 2:
            v_reference = 0.0002687502798474384
            u_reference = 0.0008030767036222218
    
        # Compute errors
        v_diff = abs(v_error - v_reference)
        u_diff = abs(u_error - u_reference)
        tolerance = 1.e-9
        msg = "Maximal %s value does not match reference: diff is %.16e"
        print(f"level: {level}")
        print("v_error = %.16e" % v_error)
        print("u_error = %.16e" % u_error)
        assert (v_diff < tolerance), msg % ("v", v_diff)
        assert (u_diff < tolerance), msg % ("u", u_diff)

@slow
@pytest.mark.xfail(dolfin_version == "2016.2.0", reason="Unknown")
def test_spatial_and_temporal_convergence():
    "Test convergence rates for bidomain solver."
    v_errors = []
    u_errors = []
    dts = []
    hs = []
    T = 0.1
    dt = 0.01
    theta = 0.5
    N = 10
    for level in (1, 2):
        a = dt/(2**level)
        v_error, u_error, h, a, T = main(N*(2**level), a, T, theta)
        v_errors.append(v_error)
        u_errors.append(u_error)
        dts.append(a)
        hs.append(h)

    v_rates = convergence_rate(hs, v_errors)
    u_rates = convergence_rate(hs, u_errors)
    print("v_errors = ", v_errors)
    print("u_errors = ", u_errors)
    print("v_rates = ", v_rates)
    print("u_rates = ", u_rates)

    assert all(v > 1.9 for v in v_rates), "Failed convergence for v"
    assert all(u > 1.9 for u in u_rates), "Failed convergence for u"


if __name__ == "__main__":
    test_analytic_bidomain()
    test_spatial_and_temporal_convergence()
