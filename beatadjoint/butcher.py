"""
Implementation of a Butcher table -> UFL system
for flexibly discretising ODEs in Dolfin.
"""

import numpy
from dolfinimport import *
import ufl
        
class ButcherTable(object):
    def __init__(self, a, b, c, name="UnknownScheme", order="UnknownOrder"):
        """
        a - numpy array for a matrix of Butcher table.
        b - numpy array for linear combination coefficients.
        c - numpy array for time evaluation offsets.
        """

        self.name = name
        self.order = order

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

    def __str__(self):
        return self.name
    
    def human_form(self):
        output = []
        for i in range(self.s):
            kterm = " + ".join("%sh*k_%s" % ("" if self.a[i,j] == 1.0 else \
                                             "%s*"%self.a[i,j], j) \
                               for j in range(self.s) if self.a[i,j] != 0)
            if self.c[i] in [0.0, 1.0]:
                cih = " + h" if self.c[i] == 1.0 else ""
            else:
                cih = " + %s*h"%self.c[i]
                
            if len(kterm) == 0:
                output.append("k_%(i)s = f(t_n%(cih)s, y_n)" % {"i": i, "cih": cih})
            else:
                output.append("k_%(i)s = f(t_n%(cih)s, y_n + %(kterm)s)" % \
                              {"i": i, "cih": cih, "kterm": kterm})

        parentheses = "(%s)" if numpy.sum(self.b>0) > 1 else "%s"
        output.append("y_{n+1} = y_n + h*" + parentheses % (" + ".join(\
            "%sk_%s" % ("" if self.b[i] == 1.0 else "%s*"%self.b[i], i) \
            for i in range(self.s) if self.b[i] > 0)))
        return "\n".join(output)

    def to_ufl(self, f, solution, time, dt):
        """
        Return a list of ufl rhs_expressions corresponding to the steps
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

        rhs_expressions = []
        k = [Function(Y, name="k_%s" % i) for i in range(self.s)]
        for i in range(self.s):

            if t is not None:
                evaltime = t + self.c[i] * dt
            else:
                evaltime = None

            evalargs = y_ + Constant(dt) * sum([float(self.a[i,j]) * k[j] \
                                                for j in range(i+1)], zero(*y_.shape()))
            equation = inner(g(evalargs, evaltime), v)*dx
            rhs_expressions.append(equation)

        y_next = Function(Y, name="y_next")
        equation = y_ + sum((float(dt*self.b[i]) * k[i] for i in range(self.s)), \
                            zero(*y_.shape()))
        
        rhs_expressions.append(equation)
        k.append(y_next)

        return (rhs_expressions, k, v)

# For those who prefer pretentious foreign words over stout Anglo-Saxon ones
ButcherTableau = ButcherTable

class ForwardEuler(ButcherTable):
    def __init__(self):
        ButcherTable.__init__(self, numpy.array([[0]]), numpy.array([1]),
                              numpy.array([0]), name="ForwardEuler", order=1)

class BackwardEuler(ButcherTable):
    def __init__(self):
        ButcherTable.__init__(self, numpy.array([[1]]), numpy.array([1]),
                              numpy.array([1]), name="BackwardEuler", order=1)

class MidpointMethod(ButcherTable):
    def __init__(self):
        ButcherTable.__init__(self, numpy.array([[0, 0], [0.5, 0]]),
                              numpy.array([0, 1]), numpy.array([0, 0.5]),
                              name="MidpointMethod",
                              order=2)

class RK4(ButcherTable):
    def __init__(self):
        ButcherTable.__init__(self, numpy.array([[0, 0, 0, 0],
                                                 [0.5, 0, 0, 0],
                                                 [0, 0.5, 0, 0],
                                                 [0, 0, 1, 0]]),
                              numpy.array([1./6, 1./3, 1./3, 1./6]),
                              numpy.array([0, 0.5, 0.5, 1]),
                              name="RungeKutta4",
                              order=4)

class NaiveODESolver(object):
    def __init__(self, rhs_expressions, variables, v):

        expressions = []

        # Check if rhs are explicit, not dependent on the corresponding
        # prognostic variable
        for rhs, variable in zip(rhs_expressions, variables):

            if isinstance(rhs, ufl.Form):
                rhs_der = ufl.algorithms.expand_derivatives(\
                    derivative(rhs, variable))
                
                # If no integrals in differentiated rhs we have an explicit step
                if not rhs_der.integrals():
                    # Explicit step
                    expressions.append(rhs)
                else:
                    form = rhs-inner(variable,v)*dx
                    expressions.append(form == 0)
            
            else:
                expressions.append(rhs)
        
        self.expressions = expressions
        self.variables = variables

    def solve(self, verbose=True):
        for expr, var in zip(self.expressions, self.variables):

            if isinstance(expr, ufl.classes.Equation):
                # If implicit make a solve
                solve(expr, var)
            elif isinstance(expr, ufl.Form):
                # If explicit just assemble the expression
                assemble(expr, tensor=var.vector())
            else:
                var.assign(expr)
            
            if verbose: print "%s: %s" % (var, var.vector().array())

if __name__ == "__main__":
    import sys
    set_log_level(ERROR)

    schemes = [ForwardEuler(), BackwardEuler(), MidpointMethod(), RK4()]

    mesh = UnitIntervalMesh(2)

    # Scalar test
    V = FunctionSpace(mesh, "R", 0)
    y = Function(V, name="y")

    # solve the ODE
    # \dot{y} = y
    form = y

    for scheme in schemes:
        print "-" * 80
        print scheme, "(Scalar)"
        print scheme.human_form()
        print "-" * 80
        y_true = Expression("exp(t)", t=1.0)
        y_errors = []

        for dt in [0.05, 0.025, 0.0125]:
          y.interpolate(Constant(1.0))
          to_ufl = scheme.to_ufl(form, y, time=Constant(0.0), dt=dt)
          solver = NaiveODESolver(*to_ufl)
          y_next = to_ufl[1][-1]

          for i in range(int(1.0/dt)):
            solver.solve(verbose=False)
            y.assign(y_next)

          y_errors.append(y_true(0.0) - y(0.0))

        print "y_errors: ", y_errors
        print "y_convergence: ", convergence_order(y_errors)
        assert min(convergence_order(y_errors)) > float(scheme.order) - 0.1

    # Vector test
    V = VectorFunctionSpace(mesh, "R", 0, dim=2)
    y = Function(V, name="y")

    # solve the ODE
    # \dot{u} = -v
    # \dot{v} =  u
    form = as_vector((-y[1], y[0]))

    for scheme in schemes:
        print "-" * 80
        print scheme, "(Vector)"
        print scheme.human_form()
        print "-" * 80
        y_true = Expression(("cos(t)", "sin(t)"), t=1.0)
        u_errors = []
        v_errors = []

        for dt in [0.05, 0.025, 0.0125]:
          y.interpolate(Expression(("1", "0")))
          to_ufl = scheme.to_ufl(form, y, time=Constant(0.0), dt=dt)
          solver = NaiveODESolver(*to_ufl)
          y_next = to_ufl[1][-1]

          for i in range(int(1.0/dt)):
            solver.solve(verbose=False)
            y.assign(y_next)

          u_errors.append((y_true(0.0) - y(0.0))[0])
          v_errors.append((y_true(0.0) - y(0.0))[1])

        print "u_errors: ", u_errors
        print "u_convergence: ", convergence_order(u_errors)
        print "v_errors: ", v_errors
        print "v_convergence: ", convergence_order(v_errors)
        assert min(convergence_order(u_errors)) > float(scheme.order) - 0.1
        assert min(convergence_order(v_errors)) > float(scheme.order) - 0.1
