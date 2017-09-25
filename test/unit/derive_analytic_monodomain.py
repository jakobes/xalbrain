"""Script that deives an analytic solution to the monodomain equations. 
It is intended for use in `test_analytic_monodomain.py`.
"""


from sympy import (
    sin,
    cos,
    symbols,
    Integer,
    Rational,
    Matrix,
    diff,
    pi,
)


def underline(s):
    print("{}\n{}".format(s, "-"*len(s.__str__())))


x, y, t = symbols("x y t")
C, lam = symbols("C lam")
M_i = Integer(1)

v = sin(t)*cos(2*pi*x)*cos(2*pi*y)

anisotropy_rates = lam/(1 + lam)

grad_v = Matrix([M_i*diff(v, x), M_i*diff(v, y)])
div_grad_v = diff(grad_v[0], x) + diff(grad_v[1], y)



underline("Checking no-flux boundary conditions on top/bottom/left/right:")
y_dir = Matrix([0, 1])
flux_y = grad_v[0]*y_dir[0] + grad_v[1]*y_dir[1]

x_dir = Matrix([1, 0])
flux_x = grad_v[0]*x_dir[0] + grad_v[1]*x_dir[1]


flux_dict = {
    "top": flux_y.subs(y, 0),
    "bottom": flux_y.subs(y, 1),
    "left": flux_x.subs(x, 0),
    "right": flux_x.subs(x, 1)
}

for key in ("top", "bottom", "left", "right"):
    print("Checking no flux on {}: {}".format(key, flux_dict[key]))


underline("Deriving right-hand side of the monodomain equation")
f = diff(v, t) - anisotropy_rates*div_grad_v
print(f.simplify())
