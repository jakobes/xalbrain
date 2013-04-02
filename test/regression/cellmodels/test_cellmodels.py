"""
Regression and correctness test for OriginalFitzHughNagumo model and pure
CellSolver: compare (in eyenorm) time evolution with results from
Section 2.4.1 p. 36 in Sundnes et al, 2006 (checked 2012-09-19), and
check that maximal v/s values do not regress
"""

# Marie E. Rognes <meg@simula.no>
# Last changed: 2012-10-23

import unittest

from dolfin import *
from beatadjoint import *

class CellSolverTestCase(unittest.TestCase):

    def test_fitzhugh_nagumo(self):
        class Stimulus(Expression):
            def __init__(self, t=0.0):
                self.t = t
            def eval(self, value, x):
                if self.t >= 50 and self.t < 60:
                    v_amp = 125
                    value[0] = 0.05*v_amp
                else:
                    value[0] = 0.0

        cell = OriginalFitzHughNagumo()
        cell.stimulus = Stimulus()
        solver = CellSolver(cell)

        # Setup initial condition
        (vs_, vs) = solver.solution_fields()
        vs_.vector()[0] = -85. # Initial condition resting state
        vs_.vector()[1] = 0.

        # Initial set-up
        (T0, T) = (0, 400)
        dt = 1.0
        t0 = T0; t1 = T0 + dt

        times = []
        v_values = []
        s_values = []

        # Solve
        while (t1 <= T):
            info_blue("Solving on t = (%g, %g)" % (t0, t1))
            timestep = (t0, t1)
            times += [(t0 + t1)/2]
            tmp = solver.step(timestep, vs_)
            vs.assign(tmp)

            v_values += [vs.vector()[0]]
            s_values += [vs.vector()[1]]

            # Update
            vs_.assign(vs)
            t0 = t1; t1 = t0 + dt

        # Regression test
        v_max_reference = 2.3839115023509514e+01
        s_max_reference = 6.9925836316850834e+01
        tolerance = 1.e-14
        msg = "Maximal %s value does not match reference: diff is %.16e"
        v_diff = abs(max(v_values) - v_max_reference)
        s_diff = abs(max(s_values) - s_max_reference)
        assert (v_diff < tolerance), msg % ("v", v_diff)
        assert (s_diff < tolerance), msg % ("s", s_diff)

        # Correctness test
        import os
        if int(os.environ.get("DOLFIN_NOPLOT", 0)) != 1:
            import pylab
            pylab.plot(times, v_values, 'b*')
            pylab.plot(times, s_values, 'r-')
            pylab.show()

    def test_fitz_hugh_nagumo_modified(self):

        k = 0.00004
        Vrest = -85.
        Vthreshold = -70.
        Vpeak = 40.
        k = 0.00004
        l = 0.63
        b = 0.013

        class OriginalFitzHughNagumoModified(CardiacCellModel):
            """ODE model:

            parameters(Vrest,Vthreshold,Vpeak,k,l,b,ist)

            input(u)
            output(g)
            default_states(v=-85, w=0)

            Vrest = -85;
            Vthreshold = -70;
            Vpeak = 40;
            k = 0.00004;
            l = 0.63;
            b = 0.013;
            ist = 0.0

            v = u[0]
            w = u[1]

            g[0] =  -k*(v-Vrest)*(w+(v-Vthreshold)*(v-Vpeak))-ist;
            g[1] = l*(v-Vrest) - b*w;

            [specified by G. T. Lines Sept 22 2012]

            Note the minus sign convention here in the specification of
            I (g[0]) !!
            """

            def __init__(self):
                CardiacCellModel.__init__(self)

            def default_parameters(self):
                parameters = Parameters("OriginalFitzHughNagumoModified")
                parameters.add("Vrest", Vrest)
                parameters.add("Vthreshold", Vthreshold)
                parameters.add("Vpeak", Vpeak)
                parameters.add("k", k)
                parameters.add("l", l)
                parameters.add("b", b)
                parameters.add("ist", 0.0)
                return parameters

            def I(self, v, w):
                k = self._parameters["k"]
                Vrest = self._parameters["Vrest"]
                Vthreshold = self._parameters["Vthreshold"]
                Vpeak = self._parameters["Vpeak"]
                ist = self._parameters["ist"]
                i =  -k*(v-Vrest)*(w+(v-Vthreshold)*(v-Vpeak))-ist;
                return -i

            def F(self, v, w):
                l = self._parameters["l"]
                b = self._parameters["b"]
                Vrest = self._parameters["Vrest"]
                return l*(v-Vrest) - b*w;

            def num_states(self):
                return 1

            def __str__(self):
                return "Modified FitzHugh-Nagumo cardiac cell model"

        def _run(cell):
            solver = CellSolver(cell)

            # Setup initial condition
            (vs_, vs) = solver.solution_fields()
            vs_.vector()[0] = 30. # Non-resting state
            vs_.vector()[1] = 0.

            T = 2
            solutions = solver.solve((0, T), 0.25)
            times = []
            v_values = []
            s_values = []
            for ((t0, t1), vs) in solutions:
                times += [0.5*(t0 + t1)]
                v_values.append(vs.vector()[0])
                s_values.append(vs.vector()[1])

            return (v_values, s_values, times)

        # Try the modified one
        cell_mod = OriginalFitzHughNagumoModified()
        (v_values_mod, s_values_mod, times_mod) = _run(cell_mod)

        # Compare with our standard FitzHugh (reparametrized)
        v_amp = Vpeak - Vrest
        cell_parameters = {"c_1": k*v_amp**2, "c_2": k*v_amp, "c_3": b/l,
                           "a": (Vthreshold - Vrest)/v_amp, "b": l,
                           "v_rest": Vrest, "v_peak": Vpeak}
        cell = OriginalFitzHughNagumo(cell_parameters)
        (v_values, s_values, times) = _run(cell)

        msg = "Mismatch in %s value comparison, diff = %.16e"
        v_diff = abs(v_values[-1] - v_values_mod[-1])
        s_diff = abs(s_values[-1] - s_values_mod[-1])
        assert (v_diff < 1.e-12), msg % v_diff
        assert (s_diff < 1.e-12), msg % s_diff

        # Look at some plots
        import os
        if int(os.environ.get("DOLFIN_NOPLOT", 0)) != 1:
            import pylab
            pylab.title("FitzHugh with converted parameters")
            pylab.plot(times, v_values, 'b*')
            pylab.plot(times, s_values, 'r-')

            pylab.figure()
            pylab.title("FitzHugh modified model")
            pylab.plot(times_mod, v_values_mod, 'b*')
            pylab.plot(times_mod, s_values_mod, 'r-')
            pylab.show()



if __name__ == "__main__":
    print ""
    print "Testing cell models and solvers"
    print "-------------------------------"
    unittest.main()
