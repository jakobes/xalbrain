"""
Main driver script for running tests: runs all .py scripts in
specified subdirectories
"""

import os
import sys
import instant

testdirectories = ["no-adjoints"]#, "adjoints"]

# Command to run
command = "python %s " + " ".join(sys.argv[1:])

# Don't plot
os.environ["DOLFIN_NOPLOT"] = "1"

# Walk through all Python files in given subdirectories and run each
# script
failed = []
for test in testdirectories:
    for subdir, dirs, files in os.walk(os.path.join(os.getcwd(), test)):
        pyfiles = [filename for filename in files
                   if (os.path.splitext(filename)[1] == ".py")]

        for pyfile in pyfiles:
            print "Running test: %s" % os.path.join(pyfile)
            print "-"*80
            absfile = os.path.join(subdir, pyfile)
            fail, output = instant.get_status_output(command % absfile)

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
