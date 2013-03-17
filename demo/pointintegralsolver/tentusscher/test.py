# NOTE: This script needs the
#
#  lp:~johan-hake/dolfin/general-rk-solver
#
# dolfin branch to work
from dolfin import *
from tentusscher_2004_mcell import *
from matplotlib import pyplot as plt
plt.interactive(True)

import numpy as np
params = default_parameters(stim_start=5)
state_init = init_values()

parameters.form_compiler.optimize=False
parameters.form_compiler.cpp_optimize=True
parameters.form_compiler.quadrature_degree = 2

mesh = UnitSquareMesh(10,10)
V = VectorFunctionSpace(mesh, "CG", 1, dim=state_init.value_size())
vertex_to_dof_map = V.dofmap().vertex_to_dof_map(mesh)
u = Function(V)
u_array = u.vector().array()
time = Constant(0.0)

form = rhs(u, time, params)
tstop = 10

ind_V = 15
dt = 0.1
dt_output = 5.0
for integral, Solver, solver_str in [(dP, PointIntegralSolver, "PointIntegralSolver"),
                                     (dx, RKSolver, "RKSolver"), 
                                     ]:
    for Scheme in [BackwardEuler, 
                   #ExplicitMidPoint, RK4, ForwardEuler, 
                   #ESDIRK3, #ESDIRK4,
                   ]:

        # Init solution Function
        u.interpolate(state_init)
        u_array[vertex_to_dof_map] = u.vector().array()

        # Create a output list for plt
        output = [u_array[ind_V]]
        time_output = [0.0]
        h1, = plt.plot(time_output, output)
        plt.show()
        plt.xlim(0, tstop)
        plt.ylim(-100, 30)
        plt.draw()

        # Create Scheme and Solver
        scheme = Scheme(form*integral, u, time)
        info(scheme)
        solver = Solver(scheme)
        scheme.t().assign(0.0)

        if solver_str == "PointIntegralSolver":
            solver.parameters.newton_solver.report = False
            solver.parameters.newton_solver.iterations_to_retabulate_jacobian = 5
            solver.parameters.newton_solver.maximum_iterations = 12
            
        # Time step
        next_dt = max(min(tstop-float(scheme.t()), dt), 0.0)
        while next_dt > 0.0:
            timer = Timer("Stepping ({0})".format(solver_str))
            
            # Step solver
            solver.step(next_dt)

            # Collect plt output data
            u_array[vertex_to_dof_map] = u.vector().array()
            output.append(u_array[ind_V])
            time_output.append(float(scheme.t()))
            h1.set_xdata(time_output)
            h1.set_ydata(output)
            plt.draw()

            # Output to screen
            if (float(scheme.t()) % dt_output) < dt:
                print "t:", float(scheme.t()), "u:", u_array[ind_V]

            # Next time step
            next_dt = max(min(tstop-float(scheme.t()), dt), 0.0)

#list_timings()
