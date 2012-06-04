import os
import dolfin

tests = ["simple.py"]

for test in tests:
    os.system("python %s" % test)
