import dolfin
import dolfin_adjoint

def join(subfunctions, V, annotate=False):
    """
    Take a list of subfunctions s[i], and return the corresponding
    mixed function s = {s[0], s[1], ..., s[n]}
    """

    # Project subfunctions onto mixed space
    return dolfin_adjoint.project(dolfin.as_vector(subfunctions), V,
                                  annotate=annotate)
