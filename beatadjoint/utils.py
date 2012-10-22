"This module provides various utilities for internal use."

# Copyright (C) 2012 Marie E. Rognes (meg@simula.no)
# Use and modify at will
# Last changed: 2012-10-11

__all__ = ["join"]

import dolfin
import dolfin_adjoint

def join(subfunctions, V, annotate=False, solver_type="lu"):
    """
    Take a list of subfunctions s[i], and return the corresponding
    mixed function s = {s[0], s[1], ..., s[n]} in V

    **Optional arguments**
    * annotate (default: False)
      turn on annotation with annotate = True
    * solver_type (default: "lu")
      adjust solver_type of projection
    """

    # Project subfunctions onto mixed space
    return dolfin_adjoint.project(dolfin.as_vector(subfunctions), V,
                                  annotate=annotate, solver_type=solver_type)
