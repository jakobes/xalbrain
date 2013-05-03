"""
Basic and more advanced tests for the cell models and their forms.
In particular, the tests

* check that cardiac cell models can be initialized
* check that basic cell model methods can be called
* check that the forms (F and I) can be compiled by a form compiler
  with various optimizations
* check that the results from compiling the forms with various form
  compiler parameters match
"""

__author__ = "Marie E. Rognes (meg@simula.no), 2013"
__all__ = [""]

import unittest
import time
import numpy.linalg
from dolfin import *
from beatadjoint import supported_cell_models
from beatadjoint.utils import state_space

def _setup_rhs_form(model):
    "Helper function to setup rhs form for a given cell model."
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 1)
    S = state_space(mesh, model.num_states())
    VS = V*S
    vs = Function(VS)
    vs.assign(project(model.initial_conditions(), VS))
    (v, s) = split(vs)

    # Define the right-hand-side of the system of ODEs: Dt(u) = rhs(u)
    # Note sign of the ionic current
    (w, r) = TestFunctions(VS)
    rhs = inner(model.F(v, s), r) + inner(- model.I(v, s), w)
    form = rhs*dP
    return (vs, form)

class TestCellModelBasics(unittest.TestCase):
    "Test basic functionality for cell models."
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
    "Test that model forms compile with various optimisations."
    def setUp(self):
        "Set-up rhs forms for all cell models."

        # NB: Reducing quadrature degree
        parameters["form_compiler"]["quadrature_degree"] = 2

        self.forms = []
        self.vss = []
        self.models = []
        for Model in supported_cell_models:
            model = Model()
            (vs, form) = _setup_rhs_form(model)
            self.models += [model]
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
        fc_parameters["cpp_optimize"] = True
        fc_parameters["cpp_optimize_flags"] = " ".join(flags)
        for form in self.forms:
            f = Form(form, form_compiler_parameters=fc_parameters)

class TestCellModelFormCompilationCorrectness(unittest.TestCase):
    "Test that various compilation options gives same results"

    def test_point_integral_solver(self):
        "Compare form compilation result with and without optimizations."

        parameters["form_compiler"]["quadrature_degree"] = 2

        def _point_integral_step(model, Solver=BackwardEuler):
            (vs, form) = _setup_rhs_form(model)

            # Set-up scheme
            time = Constant(0.0)
            scheme = Solver(form, vs, time)
            scheme.t().assign(float(time))

            # Create and step solver
            solver = PointIntegralSolver(scheme)
            solver.parameters.newton_solver.maximum_iterations = 15
            #solver.parameters.newton_solver.report = False
            dt = 0.05
            solver.step(dt)
            return vs

        # For each model: compare result with and without aggressive
        # optimizations
        for Model in supported_cell_models:
            model = Model()

            tolerance = 1.e-12

            # Run with no particular optimizations
            vs = _point_integral_step(model)
            non_opt_result = vs.vector().array()

            # Turn on aggresssive optimizations
            flags = "-O3 -ffast-math -march=native"
            parameters["form_compiler"]["cpp_optimize"] = True
            parameters["form_compiler"]["cpp_optimize_flags"] = flags
            vs = _point_integral_step(model)
            opt_result = vs.vector().array()

            # Compare results
            c = non_opt_result - opt_result
            c_inf = numpy.linalg.norm(c, numpy.inf)
            print "|c|_inf = ", numpy.linalg.norm(c, numpy.inf)
            assert (c_inf < tolerance), "Mismatch in compiled results."

            # Reset parameters by turning off optimizations
            parameters["form_compiler"]["cpp_optimize"] = True
            parameters["form_compiler"]["cpp_optimize_flags"] = "-O2"

            vs = _point_integral_step(model)
            opt_result = vs.vector().array()

            # Compare results
            c = non_opt_result - opt_result
            c_inf = numpy.linalg.norm(c, numpy.inf)
            print "|c|_inf = ", numpy.linalg.norm(c, numpy.inf)
            assert (c_inf < tolerance), "Mismatch in compiled results."

            # Reset parameters by turning off optimizations
            parameters["form_compiler"]["representation"] = "uflacs"
            
            vs = _point_integral_step(model)
            opt_result = vs.vector().array()

            # Compare results
            c = non_opt_result - opt_result
            c_inf = numpy.linalg.norm(c, numpy.inf)
            print "|c|_inf = ", numpy.linalg.norm(c, numpy.inf)
            assert (c_inf < tolerance), "Mismatch in compiled results."

if __name__ == "__main__":
    print("")
    print("Testing cell models")
    print("-------------------")
    unittest.main()
