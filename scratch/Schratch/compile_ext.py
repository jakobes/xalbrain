
from fenics import *
import instant
import os


source_files = [
    "adex/AdexPointIntegralSolver.h",
    "adex/AdexPointIntegralSolver.cpp",
]

def readSourceFile(filepath):
    with open(filepath, "r") as source_file:
        return source_file.read() 

code = "\n\n\n".join(map(readSourceFile, source_files))

try:
    ext_module = compile_extension_module(
        code,
        additional_system_headers=[
            "vector",
        ]
    )
    pass
except RuntimeError as e:
    with open(e.__str__().split("'")[-2]) as errfile:
        print(errfile.read())
    raise
