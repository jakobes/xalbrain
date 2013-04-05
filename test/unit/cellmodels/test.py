"""
Basic and more advanced tests for the cell models and their forms.
"""

__author__ = "Marie E. Rognes (meg@simula.no), 2013"
__all__ = [""]

import unittest
import time
import numpy.linalg
from dolfin import *
from beatadjoint import supported_cell_models, BasicSplittingSolver

class TestCellModelBasics(unittest.TestCase):
    def test_create_cell_model(self):
        "Test that all supported cell models can be initialized and printed."
        for Model in supported_cell_models:
            model = Model()
            print("Successfully created %s." % model)

    def test_create_cell_model_ics(self):
        "Test that all supported cell models have initial conditions."
        for Model in supported_cell_models:
            model = Model()
            ics = model.initial_conditions()

class TestCellModelFormCompilation(unittest.TestCase):

    def setUp(self):
        "Set-up rhs forms for all cell models."

        parameters["form_compiler"]["quadrature_degree"] = 2

        self.forms = []
        self.vss = []
        self.models = []

        for Model in supported_cell_models:
            model = Model()
            self.models += [model]

            # Set-up mesh and functions
            #mesh = UnitCubeMesh(2, 2, 2)
            mesh = UnitSquareMesh(100, 100)
            V = FunctionSpace(mesh, "CG", 1)
            S = BasicSplittingSolver.state_space(mesh, model.num_states())
            VS = V*S
            vs = Function(VS)
            vs.assign(project(model.initial_conditions(), VS))
            (v, s) = split(vs)

            # Define the right-hand-side of the system of ODEs: Dt(u) = rhs(u)
            # Note sign of the ionic current
            (w, r) = TestFunctions(VS)
            rhs = inner(model.F(v, s), r) + inner(- model.I(v, s), w)
            form = rhs*dP
            self.vss += [vs]
            self.forms += [form]

    def test_form_compilation(self):
        "Test that the forms defining the model can be compiled by FFC."
        print("Testing form compilation.")
        for (i, form) in enumerate(self.forms):
            f = Form(form)

    def test_form_compilation_cpp_optimize(self):
        """Test that the forms defining the model can be compiled by
        FFC with standard optimization."""
        print("Testing form compilation with cpp optimizations.")
        fc_parameters = parameters["form_compiler"].copy()
        fc_parameters["cpp_optimize"] = True
        for form in self.forms:
            f = Form(form, form_compiler_parameters=fc_parameters)

    def test_form_compilation_optimize(self):
        """Test that the forms defining the model can be compiled by
        FFC with FFC optimizations."""
        print("Testing form compilation with FFC optimizations.")
        fc_parameters = parameters["form_compiler"].copy()
        fc_parameters["optimize"] = True
        for (i, form) in enumerate(self.forms):
            f = Form(form, form_compiler_parameters=fc_parameters)

    def test_form_compilation_custom_optimization(self):
        """Test that the forms defining the model can be compiled by
        FFC with custom optimizations."""
        print("Testing form compilation with custom optimization.")
        flags = ["-O3", "-ffast-math", "-march=native"]
        fc_parameters = parameters["form_compiler"].copy()
        fc_parameters["cpp_optimize_flags"] = " ".join(flags)
        for form in self.forms:
            f = Form(form, form_compiler_parameters=fc_parameters)

    # FIXME: This test is not yet working. Results are very odd, and
    # something is very fishy with the parameters.
    def test_point_integral_solver(self):
        """Test that point integral solver gives same result with and
        without various optimisations."""

        def _run(form, vs, Solver):
            time = Constant(0.0)
            scheme = Solver(form, vs, time)
            scheme.t().assign(float(time))

            # Create and step solver
            solver = PointIntegralSolver(scheme)
            #solver.parameters.newton_solver.report = False
            dt = 0.1
            solver.step(dt)

        for (i, (form, vs)) in enumerate(zip(self.forms, self.vss)):

            # NB: FIXME
            if (i != 2): break

            print "\n\nTesting %s with FFC optimizations" % self.models[i]

            parameters["form_compiler"]["quadrature_degree"] = 2
            parameters["form_compiler"]["optimize"] = True
            parameters["form_compiler"]["cpp_optimize"] = True
            parameters["form_compiler"]["cpp_optimize_flags"] = "-O2"

            start = time()
            _run(form, vs, BackwardEuler)
            time_elapsed = time() - start
            print "time_elapsed = ", time_elapsed
            a = vs.vector().array()

            print "\n\nTesting %s with gcc optimizations" % self.models[i]

            parameters["form_compiler"]["optimize"] = False
            parameters["form_compiler"]["cpp_optimize"] = True
            flags = ["-O3", "-ffast-math", "-march=native"]
            parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)

            start = time()
            _run(form, vs, BackwardEuler)
            time_elapsed = time() - start
            print "time_elapsed with optimizations = ", time_elapsed

            b = vs.vector().array()
            c = a - b
            print "|c|_inf = ", numpy.linalg.norm(c, numpy.inf)

if __name__ == "__main__":
    print("")
    print("Testing cell models")
    print("-------------------")
    unittest.main()
