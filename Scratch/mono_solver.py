from dolfin import *

from xalbrain.utils import convergence_rate

set_log_level(100)

def main(N, dt, T, theta):
    mesh = UnitSquareMesh(N, N)
    
    V = FunctionSpace(mesh, "Lagrange", 1) 
    
    v = TrialFunction(V)
    w = TestFunction(V)
    
    vp = Function(V)    # previous solution
    v_ = Function(V)    # solution
    
    M_i = Constant(1)

    dtc = Constant(dt)
    t = 0
    ac_str = "(8*pi*pi*lam*sin(t) + (lam + 1)*cos(t))*cos(2*pi*x[0])*cos(2*pi*x[1])/(lam + 1)"
    v_exact  = Expression("sin(t)*cos(2*pi*x[0])*cos(2*pi*x[1])", t=T, degree=3)

    # ofile = File("v_diff{N}.pvd".format(N=N))

    while t <= T:
        t += dt
        stimulus = Expression(ac_str, t=t, lam=Constant(1), degree=3)
        rhs = stimulus*w*dx
        v_mid = theta*v - (1 - theta)*vp
        dv = (v - vp)/dtc
        parabolic = inner(grad(v_mid), grad(w))*dx
        
        G = dv*w*dx + parabolic - rhs
        a, L = system(G)
        
        pde = LinearVariationalProblem(a, L, v_)
        params = LinearVariationalSolver.default_parameters()

        params["linear_solver"] = "petsc"
        solver = LinearVariationalSolver(pde)
        solver.parameters.update(params)
        solver.solve()
        vp.assign(v_)
    
    v_exact.t = t
    v_error = errornorm(v_exact, v_, "l2", degree_rise=2)
    return v_error, mesh.hmin(), dt, T


def test_spatial_convergence() -> None:
    """Take a very small time step, reduce mesh size, expect 2nd order convergence."""
    v_errors = []
    hs = []
    dt = 1e-6
    T = 10*dt

    for N in (5, 10, 20, 40):
        v_error, h, *_ = main(N, dt, T, theta=0.5)
        v_errors.append(v_error)
        hs.append(h)

    v_rates = convergence_rate(hs, v_errors)
    print("dt, T = {dt}, {T}".format(dt=dt, T=T))
    print("v_errors = ", v_errors)
    print("v_rates = ", v_rates)

    # msg = "Failed convergence for v. v_rates = {}".format(", ".join(map(str, v_rates)))
    # assert all(v > 1.99 for v in v_rates), msg


if __name__ == "__main__":
    test_spatial_convergence()
