import sys
import os
import pytest

# Disable the xdist plugin by running py.test like this:
# py.test -p no:xdist

# Automatically parallelize over all cpus
def pytest_cmdline_preparse(args):
    if 'xdist' in sys.modules: # pytest-xdist plugin
        import multiprocessing
        num = multiprocessing.cpu_count()
        args[:] = ["-n", str(num)] + args


def pytest_runtest_setup(item):
    """ Hook function  which is called before every test """

    import beatadjoint
    # Reset dolfin parameter dictionary
    # FIXME: Is there a better way of doing this?
    beatadjoint.parameters["form_compiler"]["optimize"] = False
    beatadjoint.parameters["form_compiler"]["cpp_optimize"] = False
    beatadjoint.parameters["reorder_dofs_serial"] = True
    beatadjoint.parameters["form_compiler"]["cpp_optimize_flags"] = '-O2'
    try:
        beatadjoint.parameters["adjoint"]["stop_annotating"] = False
        beatadjoint.parameters["adjoint"]["record_all"] = True
    except KeyError:
        pass

    try:
        # Reset adjoint tape
        from beatadjoint import adj_reset
        adj_reset()
    except ImportError:
        pass
