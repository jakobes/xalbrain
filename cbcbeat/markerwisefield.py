# Copyright (C) 2014 Marie E. Rognes (meg@simula.no)
# Use and modify at will
# Last changed: 2014-11-02

__all__ = ["MarkerwiseField"]


class MarkerwiseField(object):
    """A container class representing a function defined by a number of
    function combined with a mesh function defining mesh domains and a
    map between the these.

    Example: Given (g0, g1), (2, 5) and markers, let

      g = g0 on domains marked by 2 in markers
      g = g1 on domains marked by 5 in markers

    *Arguments*
      functions (tuple of :py:class:`dolfin.GenericFunction`)
        the different functions
      keys (tuple of ints)
        a map from the functions to the domains marked in markers
      markers (:py:class:`dolfin.MeshFunction`)
        a mesh function mapping which domains the mesh consist of

    """

    def __init__(self, functions, keys, markers):
        "Create MarkerwiseField from given input."

        # Check input
        assert len(functions) == len(keys), \
            "Expecting the number of functions to equal the number of keys"

        # Store attributes:
        self._functions = functions
        self._keys = keys
        self._markers = markers

    @property
    def functions(self):
        "The functions"
        return self._functions

    @property
    def keys(self):
        "The keys or domain numbers"
        return self._keys

    @property
    def markers(self):
        "The markers"
        return self._markers


if __name__ == "__main__":

    from dolfin import *
    g1 = Expression("1.0")
    g5 = Expression("sin(pi*x[0])")

    mesh = UnitSquareMesh(16, 16)

    class SampleDomain(SubDomain):
        def inside(self, x, on_boundary):
            return all(x <= 0.5 + DOLFIN_EPS)

    markers = MeshFunction("size_t", mesh, mesh.topology().dim(), 1)
    domain = SampleDomain()
    domain.mark(markers, 5)

    g = MarkerwiseField((g1, g5), (1, 5), markers)

    plot(markers, interactive=True)
