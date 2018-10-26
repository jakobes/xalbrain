"""Basic and more advanced tests for the cell models and their forms."""

__author__ = "Marie E. Rognes (meg@simula.no), 2013 -- 2014"

import dolfin as df

import pytest

from xalbrain.utils import state_space

from xalbrain.cellmodels import (
    Adex,
    AdexManual,
    CardiacCellModel,
)

from testutils import (
    fast,
    slow,
    parametrize,
    assert_almost_equal,
)

from testutils import (
    cell_model,
    ode_test_form,
)


class TestModelCreation:
    """Test basic features of cell models."""
    def test_create_cell_model_has_ics(self, cell_model):
        """Test that cell model has initial conditions."""
        model = cell_model
        ics = model.initial_conditions()


class TestFormCompilation:
    """Test form compilation with different optimizations."""
    def test_form_compilation(self, ode_test_form):
        """Test that form can be compiled by FFC."""
        f = df.Form(ode_test_form)

    @slow
    def test_optimized_form_compilation(self, ode_test_form):
        """Test that form can be compiled by FFC with optimizations."""
        ps = df.parameters["form_compiler"].copy()
        ps["cpp_optimize"] = True
        f = df.Form(ode_test_form, form_compiler_parameters=ps)

    @slow
    def test_custom_optimized_compilation(self, ode_test_form):
        """Test that form can be compiled with custom optimizations."""
        ps = df.parameters["form_compiler"].copy()
        ps["cpp_optimize"] = True
        flags = ["-O3", "-ffast-math", "-march=native"]
        ps["cpp_optimize_flags"] = " ".join(flags)
        f = df.Form(ode_test_form, form_compiler_parameters=ps)


class TestCompilationCorrectness:
    """Test form compilation results with different optimizations."""

    def point_integral_step(self, model: CardiacCellModel, adex: bool=False) -> df.Function:
        # Set-up forms
        mesh = df.UnitSquareMesh(10, 10)
        V = df.FunctionSpace(mesh, "CG", 1)
        S = state_space(mesh, model.num_states())
        Me = df.MixedElement((V.ufl_element(), S.ufl_element()))
        VS = df.FunctionSpace(mesh, Me)
        vs = df.Function(VS)
        vs.assign(project(model.initial_conditions(), VS))
        v, s = df.split(vs)
        w, r = df.TestFunctions(VS)
        rhs = df.inner(df.model.F(v, s), r) + df.inner(-model.I(v, s), w)
        form = rhs*df.dP

        # Set-up scheme
        time = df.Constant(0.0)
        scheme = df.BackwardEuler(form, vs, time)
        scheme.t().assign(float(time))

        # Create and step solver
        if adex:
            solver = AdexPointIntegralSolver(scheme)
        else: 
            solver = PointIntegralSolver(scheme)
        solver.parameters["newton_solver"]["relative_tolerance"] = 1e-6
        solver.parameters["newton_solver"]["report"] = False
        dt = 0.1
        solver.step(dt)
        return vs

    @slow
    @pytest.mark.parametrize("adex_model, adex", [
        pytest.param(Adex, True, marks=pytest.mark.xfail),
        pytest.param(AdexManual, False, marks=pytest.mark.xfail)
    ])
    def test_point_integral_solver(self, adex_model, adex):
        """Compare form compilation result with and without optimizations."""

        df.parameters["form_compiler"]["representation"] = "quadrature"
        df.parameters["form_compiler"]["quadrature_degree"] = 2
        tolerance = 1e-12

        # Run with no particular optimizations
        vs = self.point_integral_step(adex_model, adex)
        non_opt_result = vs.vector().array()

        # Compare with results using aggresssive optimizations
        flags = "-O3 -ffast-math -march=native"
        df.parameters["form_compiler"]["cpp_optimize"] = True
        df.parameters["form_compiler"]["cpp_optimize_flags"] = flags
        vs = self.point_integral_step(adex_model, adex)
        assert_almost_equal(non_opt_result, vs.vector().array(), tolerance)

        # Compare with results using standard optimizations
        df.parameters["form_compiler"]["cpp_optimize"] = True
        df.parameters["form_compiler"]["cpp_optimize_flags"] = "-O2"
        vs = self.point_integral_step(adex_model, adex)
        assert_almost_equal(non_opt_result, vs.vector().array(), tolerance)

        # Compare with results using uflacs if installed
        try:
            df.parameters["form_compiler"]["representation"] = "uflacs"
            vs = self.point_integral_step(adex_model, adex)
            assert_almost_equal(non_opt_result, vs.vector().array(), tolerance)
        except:
            pass

        # Reset parameters
        df.parameters["form_compiler"]["representation"] = "auto"
