import os
import sys
import instant

tests = ["simple", "test_bumps", "analytic_bidomain", "cellmodeltest"]

failed = []

# Command to run
command = "python %s.py " + " ".join(sys.argv[1:])

os.environ["DOLFIN_NOPLOT"] = "1"

for test in tests:
    if not os.path.isfile(os.path.join("%s.py" % test)):
        continue
    print "Running tests: %s" % test
    print "----------------------------------------------------------------------"
    fail, output = instant.get_status_output(command % test)
    if fail:
        print "failed"
        failed.append(test)
    else:
        print "OK"

    print ""

if failed:
    print "Failed:", ", ".join(failed)
    sys.exit(len(failed))

print "All tests OK"
