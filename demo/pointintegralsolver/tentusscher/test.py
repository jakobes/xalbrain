from dolfin import *
from tentusscher_2004_mcell import *
from matplotlib import pyplot as plt
plt.interactive(True)

import numpy as np
params = default_parameters(stim_start=1)
state_init = init_values()

parameters.form_compiler.optimize=False
#parameters.form_compiler.cpp_optimize=True
parameters.form_compiler.quadrature_degree = 2
parameters["form_compiler"]["cpp_optimize"] = True

opt = True

if opt:
    parameters["form_compiler"]["cpp_optimize_flags"] = "-O3 -ffast-math -march=native"
#parameters.form_compiler.representation = "uflacs"

mesh = UnitSquareMesh(10,10)
V = VectorFunctionSpace(mesh, "CG", 1, dim=state_init.value_size())
vertex_to_dof_map = V.dofmap().vertex_to_dof_map(mesh)
time = Constant(0.0)

tstop = 10
ind_V = 15
dt_org = 0.1
dt_output = 5.0
Vm_reference = np.fromfile("Vm_reference.npy")
dt_ref = 0.1
time_ref = np.linspace(0,tstop, int(tstop/dt_ref)+1)

for Scheme in [BackwardEuler, 
               #CN2,
               #ExplicitMidPoint,
               #RK4,
               #ForwardEuler, 
               #ESDIRK3,
               #ESDIRK4,
               ]:

    plt.clf()

    # Init solution Function
    u = Function(V)
    u.interpolate(state_init)
    u_array = np.zeros(mesh.num_vertices()*V.dofmap().num_entity_dofs(0), dtype=np.float_)
    u_array[vertex_to_dof_map] = u.vector().array()

    # Get form
    form = rhs(u, time, params)

    # Create a output list for plt
    output = [u_array[ind_V]]
    time_output = [0.0]
    plt.plot(time_ref, Vm_reference)
    h1, = plt.plot(time_output, output)
    plt.show()
    plt.xlim(0, tstop)
    plt.ylim(-100, 40)
    plt.draw()

    # Create Scheme and Solver
    scheme = Scheme(form*dP, u, time)
    info(scheme)
    solver = PointIntegralSolver(scheme)
    scheme.t().assign(0.0)
    plt.legend(["CellML reference", "gotran generated ufl model ({0})".format(\
        scheme)], "lower right")

    solver.parameters.newton_solver.report = False
    
    dt = dt_org/2 if isinstance(scheme, BackwardEuler) else dt_org

    # Time step
    next_dt = max(min(tstop-float(scheme.t()), dt), 0.0)
    while next_dt > 0.0:
        timer = Timer("Stepping ({0})".format(scheme))
        
        # Step solver
        solver.step(next_dt)
        timer.stop()

        # Collect plt output data
        u_array[vertex_to_dof_map] = u.vector().array()
        output.append(u_array[ind_V])
        time_output.append(float(scheme.t()))
        h1.set_xdata(time_output)
        h1.set_ydata(output)
        plt.draw()

        # Next time step
        next_dt = max(min(tstop-float(scheme.t()), dt), 0.0)

    output = np.array(output)
    # Compare solution from CellML run using opencell
    print scheme
    print "V[-1] = ", output[-1]
    print "V_ref[-1] = ", Vm_reference[-1]
    if not isinstance(scheme, BackwardEuler):
        offset = len(output)-len(Vm_reference)
        print "|(V-V_ref)/V_ref| = ", np.sqrt(np.sum(((\
            Vm_reference-output[:-offset])/Vm_reference)**2))/len(Vm_reference)

list_timings()
