from dolfin import *
import numpy as np
from itertools import chain




code = """
namespace dolfin {
    void repolarise(std::shared_ptr<Function>);
    
    void repolarise(std::shared_ptr<Function> solution) {
        solution->function_space()->sub(0)->dofmap(); 
    }
}
"""


try:
    ext_module = compile_extension_module(
        code,
        additional_system_headers=[
            "vector",
            "petscvec.h",
        ]
    )
    pass
except RuntimeError as e:
    with open(e.__str__().split("'")[-2]) as errfile:
        print(errfile.read())
    raise


mesh = UnitIntervalMesh(5)

V = FiniteElement("CG", mesh.ufl_cell(), 1)
W = FiniteElement("CG", mesh.ufl_cell(), 1)
Z = MixedElement((V, W))

U = FunctionSpace(mesh, Z)
u = Function(U)

ext_module.repolarise(u)

# module.repolarise
