from dolfin_utils.pjobs import submit

# Define name for job
name = "heart-simulation"

# Define job
job = "python demo.py"

# Submit job
print "Submitting %s with name %s" % (job, name)
submit(job, nodes=4, ppn=8, keep_environment=True, name=name,
       walltime=24*5)
