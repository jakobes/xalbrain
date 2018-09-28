import dolfin as df
import numpy as np
import xalbrain as xb


def splat(vs, dim):
    if vs.function_space().ufl_element().num_sub_elements() == dim:
        v = vs[0]
        if dim == 2:
            s = vs[1]
        else:
            s = df.as_vector([vs[i] for i in range(1, dim)])
    else:
        v, s = df.split(vs)
    return v, s


def get_mesh(N) -> df.Mesh:
    """Create the mesh."""
    mesh = df.UnitSquareMesh(N, N)
    return mesh


def get_cell_function(mesh: df.Mesh) -> df.MeshFunction:
    """Return a cell function intended for multi cell model."""
    mesh_function = df.MeshFunction("size_t", mesh, mesh.geometry().dim())
    mesh_function.set_all(1)
    # df.CompiledSubDomain("x[0] < 0.05").mark(mesh_function, 12)
    df.CompiledSubDomain("x[0] > 5.0").mark(mesh_function, 12)
    return mesh_function


def get_models(cell_function):
    Cressman = xb.cellmodels.Cressman
    cressman_parameters = Cressman.default_parameters()
    cressman_parameters["Koinf"] = 4
    cressman_k4 = Cressman(params=cressman_parameters)

    cressman_parameters["Koinf"] = 8
    cressman_k8 = Cressman(params=cressman_parameters)

    c0 = xb.cellmodels.Wei()
    c1 = xb.cellmodels.FitzHughNagumoManual()
    c2 = xb.cellmodels.Cressman()

    # return c2
    multi_cell_model = xb.cellmodels.MultiCellModel(
        (c2, c2),
        # (c2, c2),
        # (cressman_k4, cressman_k8),
        # (cressman_k4, cressman_k4),
        # (cressman_k4,),
        (1, 12),
        cell_function
    )
    return multi_cell_model


def solver(interval, dt=0.1, theta=1):
    t0, t1 = interval

    mesh = get_mesh(10)
    cell_function = get_cell_function(mesh)
    model = get_models(cell_function)
    num_global_states = model.num_states()

    # Create time keepers and time step
    const_dt = df.Constant(dt)
    current_time = df.Constant(0)
    t = t0 + theta*(t1 - t0)
    current_time.assign(t)

    # Create stimulus
    stimulus = df.Constant(0)

    # Make shared function pace
    mixed_vector_function_space = df.VectorFunctionSpace(
        mesh,
        "DG",
        0,
        dim=num_global_states + 1
    )

    # Create previous solution and assign initial conditions
    previous_solution = df.Function(mixed_vector_function_space)

    # _model = xb.cellmodels.Cressman()
    # previous_solution.assign(_model.initial_conditions())        # Initial condition
    previous_solution.assign(model.initial_conditions())        # Initial condition

    v_previous, s_previous = splat(previous_solution, num_global_states + 1)

    # Create current solution
    current_solution = df.Function(mixed_vector_function_space)
    v_current, s_current = splat(current_solution, num_global_states + 1)

    # Create test functions
    test_functions = df.TestFunction(mixed_vector_function_space)
    test_v, test_s = splat(test_functions, num_global_states + 1)

    # Crate time derivatives
    Dt_v = (v_current - v_previous)/const_dt
    Dt_s = (s_current - s_previous)/const_dt

    # Create midpoint evaluations following theta rule
    v_mid = theta*v_current + (1.0 - theta)*v_previous
    s_mid = theta*s_current + (1.0 - theta)*s_previous

    if isinstance(model, xb.MultiCellModel):
        # model = xb.cellmodels.Cressman()
        # dy = df.Measure("dx", domain=mesh)
        # F_theta = model.F(v_mid, s_mid, time=current_time)
        # I_theta = -model.I(v_mid, s_mid, time=current_time)
        # lhs = (Dt_v - I_theta)*test_v
        # lhs += df.inner(Dt_s - F_theta, test_s)
        # lhs *= dy()



        dy = df.Measure("dx", domain=mesh, subdomain_data=model.markers())
        domain_indices = model.keys()
        lhs_list = list()
        for k, model_k in enumerate(model.models()):
            domain_index_k = domain_indices[k]
            F_theta = model.F(v_mid, s_mid, time=current_time, index=domain_index_k)
            I_theta = -model.I(v_mid, s_mid, time=current_time, index=domain_index_k)
            a = (Dt_v - I_theta)*test_v
            a += df.inner(Dt_s - F_theta, test_s)
            a *= dy(domain_index_k)
            lhs_list.append(a)
        # Sum the form
        lhs = sum(lhs_list)
    else:
        # Evaluate currents at averaged v and s. Note sign for I_theta
        model = xb.cellmodels.Cressman()
        dy = df.Measure("dx", domain=mesh)
        F_theta = model.F(v_mid, s_mid, time=current_time)
        I_theta = -model.I(v_mid, s_mid, time=current_time)
        lhs = (Dt_v - I_theta)*test_v
        lhs += df.inner(Dt_s - F_theta, test_s)
        lhs *= dy()
    rhs = stimulus*test_v*dy()
    G = lhs - rhs

    # Solve system
    current_solution.assign(previous_solution)
    pde = df.NonlinearVariationalProblem(
        G,
        current_solution,
        J=df.derivative(G, current_solution)
    )
    solver = df.NonlinearVariationalSolver(pde)
    solver.solve()


if __name__ == "__main__":
    dt = 0.01
    T = 10*dt
    solver((0, T), dt=dt, theta=0.5)
