"""
"""

__author__ = "Marie E. Rognes (meg@simula.no), 2013"
__all__ = [""]

import unittest
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

        self.forms = []

        for Model in supported_cell_models:
            model = Model()

            # Set-up mesh and functions
            mesh = UnitCubeMesh(2, 2, 2)
            V = FunctionSpace(mesh, "CG", 1)
            S = BasicSplittingSolver.state_space(mesh, model.num_states())
            VS = V*S
            vs = Function(VS)
            (v, s) = split(vs)

            # Define the right-hand-side of the system of ODEs: Dt(u) = rhs(u)
            # Note sign of the ionic current
            (w, r) = TestFunctions(VS)
            rhs = inner(model.F(v, s), r) + inner(- model.I(v, s), w)
            form = rhs*dP
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
        for form in self.forms:
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


if __name__ == "__main__":
    print("")
    print("Testing cell models")
    print("-------------------")
    unittest.main()
