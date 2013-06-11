"""
Main driver script for running tests: runs all test_*.py scripts in
specified subdirectories
"""

# Marie E. Rognes (meg@simula.no), 2012--2013

import os
import sys
import instant

default_test_directories = ["unit", "system", "regression"]

def pretty_print(msg):
    blocks = "-"*len(msg)
    print(msg)
    print(blocks)

curdir = os.path.abspath(os.path.curdir)

# Parse command-line arguments for test_directories
test_directories = default_test_directories
if len(sys.argv) > 1:
    test_directories = sys.argv[1:]

print "Running tests in %r" % test_directories

# Command to run
command = "python %s " + " ".join(sys.argv[1:])

# Don't plot
os.environ["DOLFIN_NOPLOT"] = "1"

# Walk through all Python files in given subdirectories and run each
# script
failed = []
for test in test_directories:
    pretty_print("Running tests in %s" % test)
    for subdir, dirs, files in os.walk(os.path.join(os.getcwd(), test)):
        pyfiles = [filename for filename in files
                   if (os.path.splitext(filename)[1] == ".py"
                       and filename[0:5] == "test_")]

        for pyfile in pyfiles:
            pretty_print("Running: %s" % os.path.join(subdir, pyfile))
            os.chdir(subdir)
            fail, output = instant.get_status_output(command % pyfile)
            os.chdir(curdir)

            if fail:
                failed.append(pyfile)
                print "failed"
                print output
            else:
                print "OK"
            print ""

if failed:
    print "Failed:", ", ".join(failed)
    sys.exit(len(failed))

print "All tests OK"
