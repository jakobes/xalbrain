"Various utilies, fixtures and marks intended for test functionality."

__author__ = "Marie E. Rognes (meg@simula.no), 2014"

from xalbrain.dolfinimport import *

import numpy.linalg
import pytest

from xalbrain.dolfinimport import parameters
from xalbrain.cellmodels import *
from xalbrain.utils import state_space

from typing import (
    Any
)

# TODO: import these in the tests
# Marks
fast = pytest.mark.fast
medium = pytest.mark.medium
slow = pytest.mark.slow
adjoint = pytest.mark.adjoint
parametrize = pytest.mark.parametrize
disabled = pytest.mark.disabled
xfail = pytest.mark.xfail


# Assertions
def assert_almost_equal(a: float, b: float, tolerance: float) -> None:
    c = a - b
    try:
        assert abs(float(c)) < tolerance
    except TypeError:
        c_inf = numpy.linalg.norm(c, numpy.inf)
        assert c_inf < tolerance


def assert_equal(a: float, b: float) -> None:
    assert a == b


def assert_true(a: Any) -> None:
    assert a is True


def assert_greater(a: float, b: float) -> None:
    assert a > b


# TODO: 
supported_cell_models_str = [
    Model.__name__ for Model in (
        FitzHughNagumoManual,
        Tentusscher_2004_mcell,
        NoCellModel
    )
]


@pytest.fixture(params=supported_cell_models_str)
def cell_model(request):
    Model = eval(request.param)
    return Model()


@pytest.fixture(params=supported_cell_models_str)
def ode_test_form(request):
    Model = eval(request.param)
    model = Model()
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 1)
    S = state_space(mesh, model.num_states())
    Mx = MixedElement((V.ufl_element(), S.ufl_element()))
    VS = FunctionSpace(mesh, Mx)
    vs = Function(VS)
    vs.assign(project(model.initial_conditions(), VS))
    (v, s) = split(vs)
    (w, r) = TestFunctions(VS)
    rhs = inner(model.F(v, s), r) + inner(- model.I(v, s), w)
    form = rhs*dP
    return form
