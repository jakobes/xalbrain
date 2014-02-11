"Various utilies, fixtures and marks intended for test functionality."

__author__ = "Marie E. Rognes (meg@simula.no), 2014"

from dolfin import *
import numpy.linalg
import pytest
from beatadjoint import supported_cell_models
from beatadjoint.utils import state_space

# Marks
fast = pytest.mark.fast
slow = pytest.mark.slow
parametrize = pytest.mark.parametrize

def assert_almost_equal(a, b, tolerance):
    c = a - b
    if type(c) in (int, float):
        assert abs(c) < tolerance
    else:
        c_inf = numpy.linalg.norm(c, numpy.inf)
        assert c_inf < tolerance

def assert_equal(a, b):
    assert a == b

def assert_true(a):
    assert a == True

def assert_greater(a, b):
    assert a > b

# Fixtures
@pytest.fixture(params=supported_cell_models)
def cell_model(request):
    Model = request.param
    return Model()

@pytest.fixture(params=supported_cell_models)
def ode_test_form(request):
    Model = request.param
    model = Model()
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 1)
    S = state_space(mesh, model.num_states())
    VS = V*S
    vs = Function(VS)
    vs.assign(project(model.initial_conditions(), VS))
    (v, s) = split(vs)
    (w, r) = TestFunctions(VS)
    rhs = inner(model.F(v, s), r) + inner(- model.I(v, s), w)
    form = rhs*dP
    return form

