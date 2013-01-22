"""
Implementation of a Butcher table -> UFL system
for flexibly discretising ODEs in Dolfin.
"""

import numpy
from dolfin import *

class ButcherTable(object):
    def __init__(self, a, b, c):
        """
        a - numpy array for a matrix of Butcher table.
        b - numpy array for linear combination coefficients.
        c - numpy array for time evaluation offsets.
        """

        self.a = a
        self.b = b
        self.c = c
        self.s = a.shape[0]
        assert self.s == a.shape[1]
        assert self.s == b.shape[0]
        assert self.s == c.shape[0]
        for i in range(self.s):
            for j in range(i):
                if a[j, i] != 0:
                    raise AssertionError("Current states cannot depend on future states!")

    def human_form(self):
        for i in range(self.s):
            kterm = " + ".join("h*%s*k_%s" % (self.a[i,j], j) \
                               for j in range(self.s) if self.a[i,j] != 0)
            if len(kterm) == 0:
                print "k_%(i)s = f(t_n + %(c)s*h, y_n)" % {"i": i, "c": self.c[i]}
            else:
                print "k_%(i)s = f(t_n + %(c)s*h, y_n + %(kterm)s)" % \
                      {"i": i, "c": self.c[i], "kterm": kterm}

        parentheses = "(%s)" if numpy.sum(self.b>0) > 1 else "%s"
        print "y_{n+1} = y_n + h*" + parentheses % (" + ".join(\
            "%s*k_%s" % (self.b[i], i) for i in range(self.s) if self.b[i] > 0))

    def to_ufl(self, f, solution, time, dt):
        """
        Return a list of ufl equations corresponding to the steps
        associated with this forward temporal discretisation.

        f - UFL form of the right hand side for the ODE dy/dt = f(t, y)
        solution - the Function appearing in f that represents the solution
        time -- the Constant appearing in f that represents time
        """

        y_ = solution
        t_ = time          # as a Constant

        try:
          t  = float(time)   # as a float
        except:
          t = None

        dt = float(dt)
        Y  = y_.function_space()
        v  = TestFunction(Y)

        def g(y, t):
            if t is not None:
              return replace(f, {y_: y, t_: t})
            else:
              return replace(f, {y_: y})

        equations = []
        k = [Function(Y) for i in range(self.s)]
        for i in range(self.s):

            if t is not None:
                evaltime = t + self.c[i] * dt
            else:
                evaltime = None

            evalargs = y_ + dt * sum(self.a[i,j] * k[j] for j in range(i))
            equation = inner(k[i], v)*dx - inner(g(evalargs, evaltime), v)*dx
            equations.append(equation)

        y_next = Function(Y)
        equation = inner(y_, v)*dx - dt*inner(sum(self.b[i] * k[i] \
                                               for i in range(self.s)), v)*dx
        equations.append(equation)

        return (equations, k, y_next)

class ForwardEuler(ButcherTable):
    def __init__(self):
        ButcherTable.__init__(self, numpy.array([[0]]), numpy.array([1]),
                              numpy.array([0]))

class BackwardEuler(ButcherTable):
    def __init__(self):
        ButcherTable.__init__(self, numpy.array([[1]]), numpy.array([1]),
                              numpy.array([1]))

class MidpointMethod(ButcherTable):
    def __init__(self):
        ButcherTable.__init__(self, numpy.array([[0, 0], [0.5, 0]]),
                              numpy.array([0, 1]), numpy.array([0, 0.5]))

class RK4(ButcherTable):
    def __init__(self):
        ButcherTable.__init__(self, numpy.array([[0, 0, 0, 0],
                                                 [0.5, 0, 0, 0],
                                                 [0, 0.5, 0, 0],
                                                 [0, 0, 1, 0]]),
                              numpy.array([1./6, 1./3, 1./3, 1./6]),
                              numpy.array([0, 0.5, 0.5, 1]))

if __name__ == "__main__":
    mesh = UnitIntervalMesh(10)
    V = FunctionSpace(mesh, "R", 0)
    y = Function(V)
    form = y

    print "-" * 80
    print "Midpoint rule: "
    print "-" * 80
    MidpointMethod().human_form()

    print "-" * 80
    print "RK4: "
    print "-" * 80
    RK4().human_form()

    print "-" * 80
    print "Backward Euler: "
    print "-" * 80
    backward = BackwardEuler()
    backward.human_form()

    to_ufl = backward.to_ufl(form, y, time=None, dt=0.1)
    print to_ufl[0]
