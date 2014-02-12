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
    try:
        from beatadjoint import adj_reset
        adj_reset()
    except ImportError:
        pass
