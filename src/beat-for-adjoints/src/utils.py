from dolfin import project, as_vector, MixedFunctionSpace
from dolfin_adjoint import project

def merge(subfunctions, V=None, annotate=False):
    # Create mixed space from components if none is given
    if V is None:
        Vs = []
        for s in subfunctions:
            V = s.function_space()
            if V.component:
                V = V.collapse()
            Vs += [V]
        V = MixedFunctionSpace(*Vs)

    # Project subfunctions onto mixed space
    return project(as_vector(subfunctions), V, annotate=annotate)


