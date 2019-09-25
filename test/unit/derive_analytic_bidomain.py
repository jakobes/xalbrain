"""
Script that derives an analytic solution to the bidomain equations --
used in test_analytic_bidomain.py.
"""

# Marie E. Rognes <meg@simula.no>
# Last changed: 2013-04-02

import sympy as sp


def derive_rhs():
    def underline(s): print(s + "\n" + "-"*len(s))

    # Declare symbols
    x, y, t = sp.symbols("x y t")

    M_i = sp.Integer(1)
    M_e = sp.Integer(1)

    v = sp.sin(t)*sp.cos(2*sp.pi*x)*sp.cos(2*sp.pi*y)
    u = - sp.Rational(1, 2)*v

    underline("Analytic solutions")
    print("v = ", v)
    print("u = ", u)
    print()

    # Compute gradients
    grad_v = sp.Matrix([sp.diff(v, x), sp.diff(v, y)])
    grad_u = sp.Matrix([sp.diff(u, x), sp.diff(u, y)])

    # Compute fluxes
    J_i = sp.Matrix([M_i*grad_v[0] + M_i*grad_u[0],
                     M_i*grad_v[1] + M_i*grad_u[1]])
    J_m = sp.Matrix([M_i*grad_v[0] + (M_i + M_e)*grad_u[0],
                     M_i*grad_v[1] + (M_i + M_e)*grad_u[1]])

    div_J_i = sp.diff(J_i[0], x) + sp.diff(J_i[1], y)
    div_J_m = sp.diff(J_m[0], x) + sp.diff(J_m[1], y)

    underline("Checking that right-hand side elliptic part is zero:")
    g =  div_J_m
    print("g = ", g)
    print()

    underline("Checking that avg(u) = 0:")
    avg_u = sp.integrate(sp.integrate(u, (x, 0, 1)), (y, 0, 1))
    print("avg(u) = ", avg_u)
    print()

    underline("Checking no-flux boundary conditions on top/bottom:")
    y_dir = sp.Matrix([0, 1])
    flux_i = J_i[0]*y_dir[0] + J_i[1]*y_dir[1]
    print("J_i * n on bottom, top = ", flux_i.subs(y, 0), ",", flux_i.subs(y, 1))
    flux_m = J_m[0]*y_dir[0] + J_m[1]*y_dir[1]
    print("J_m * n on bottom, top = ", flux_m.subs(y, 0), ",", flux_m.subs(y, 1))
    print()

    underline("Checking no-flux boundary conditions on left/right:")
    x_dir = sp.Matrix([1, 0])
    flux_i = J_i[0]*x_dir[0] + J_i[1]*x_dir[1]
    print("J_i * n on left, right = ", flux_i.subs(x, 0), ",", flux_i.subs(x, 1))
    flux_m = J_m[0]*x_dir[0] + J_m[1]*x_dir[1]
    print("J_m * n on left, right = ", flux_m.subs(x, 0), ",", flux_m.subs(x, 1))
    print()

    underline("Deriving right-hand side for (parabolic) bidomain equation")
    f = sp.diff(v, t) - div_J_i
    print("f = ", f)
    print()


if __name__ == "__main__":
    derive_rhs()
