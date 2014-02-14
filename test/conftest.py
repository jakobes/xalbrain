import sys
import os
import pytest
import beatadjoint

# Disable the xdist plugin by running py.test like this:
# py.test -p no:xdist

# Automatically parallelize over all cpus
def pytest_cmdline_preparse(args):
    if 'xdist' in sys.modules: # pytest-xdist plugin
        import multiprocessing
        num = multiprocessing.cpu_count()
        args[:] = ["-n", str(num)] + args


default_params = beatadjoint.parameters.copy()
def pytest_runtest_setup(item):
    """ Hook function which is called before every test """

    # Reset dolfin parameter dictionary
    beatadjoint.parameters.update(default_params)

    # Reset adjoint state
    if beatadjoint.dolfin_adjoint:
        beatadjoint.adj_reset()
