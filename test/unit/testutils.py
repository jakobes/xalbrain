"Various utilies, fixtures and marks intended for test functionality."

__author__ = "Marie E. Rognes (meg@simula.no), 2014"

import dolfin as df

import numpy.linalg
import pytest

from xalbrain.cellmodels import *
from xalbrain.utils import state_space

from xalbrain.cellmodels import (
    SUPPORTED_CELL_MODELS,
)

from typing import (
    Any
)

SUPPORTED_CELL_MODELS_STR = list(map(lambda x: x.__name__, SUPPORTED_CELL_MODELS))


# Assertions
def assert_almost_equal(a: Any, b: Any, tolerance: float) -> None:
    """Assert that a and b are almostequal.

    numpy.linalg.norm(a -b, numpy.inf) will be used in case ofarrays.
    """
    c = a - b
    msg = "diff = {}"
    try:
        assert abs(float(c)) < tolerance, msg.format(c)
    except TypeError:
        c_inf = numpy.linalg.norm(c, numpy.inf)
        assert c_inf < tolerance, msg.format(c_inf)


def assert_equal(a: float, b: float) -> None:
    """Assert thattwo reals a and b are equal."""
    msg = "{} != {}".format(a, b) 
    assert a == b, msg


def assert_true(a: Any) -> None:
    """Assert a is True or not None.""" 
    assert a is True


def assert_greater(a: float, b: float) -> None:
    """Assert a > b."""
    msg = "{} <= {}".format(a, b)
    assert a > b, msg


@pytest.fixture(params=SUPPORTED_CELL_MODELS_STR)
def cell_model(request):
    """Eval cell model."""
    Model = eval(request.param)
    return Model()


@pytest.fixture(params=SUPPORTED_CELL_MODELS_STR)
def ode_test_form(request):
    Model = eval(request.param)
    model = Model()
    mesh = df.UnitSquareMesh(10, 10)
    V = df.FunctionSpace(mesh, "CG", 1)
    S = state_space(mesh, model.num_states())
    Mx = df.MixedElement((V.ufl_element(), S.ufl_element()))
    VS = df.FunctionSpace(mesh, Mx)
    vs = df.Function(VS)
    vs.assign(df.project(model.initial_conditions(), VS))
    (v, s) = df.split(vs)
    (w, r) = df.TestFunctions(VS)
    rhs = df.inner(model.F(v, s), r) + df.inner(- model.I(v, s), w)
    form = rhs*df.dP
    return form
