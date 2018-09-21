from dolfin import *


class BDSolver:
    def __init__(self, mesh, dt):
        self._mesh = mesh
        self._dt = Constant(dt)
        self.time = Constant(0)

    def solution_fields():
        return self.UV

    def _create_variational_forms(self):
        F_ele = FiniteElement("CG", self._mesh.ufl_cell(), 1)
        W = FunctionSpace(self._mesh, MixedElement((F_ele, F_ele)))
        u, v = TrialFunctions(W)
        w, q = TestFunctions(W)

        # Re = FiniteElement("R", self._mesh.ufl_cell(), 0)
        # W = FunctionSpace(mesh, MixedElement((F_ele, F_ele, Re)))
        # u, v, l = TrialFunctions(W)
        # w, q, lam = TestFunctions(W)

        self.UV = Function(W)
        self.UV_ = Function(W)

        Me, Mi = Constant(1), Constant(1)

        stimulus = Expression(
            # "1e-4*exp(-0.003*pow(x[0]-10, 2))*exp(-0.003*pow(x[1]-68, 2))*exp(-0.003*pow(x[2]-32.0, 2))*exp(-t)",
            "exp(-a*pow(x[0]-1, 2))*exp(-a*pow(x[1]-1, 2))*exp(-a*pow(x[2]-1, 2))*exp(-t)",
            t=self.time,
            a=0.03,
            degree=1
        )

        ect_current = Expression(
            "exp(-a*pow(x[0]-0, 2))*exp(-a*pow(x[1]-1, 2))*exp(-a*pow(x[2]-1, 2))*exp(-t)",
            t=self.time,
            a=0.03,
            degree=1
        )


        # U_, V_, _ = split(self.UV)
        U_, V_ = split(self.UV_)
        dtc = self._dt

        self.UV.vector().array()[:] = 0

        # rec_space = V_.function_space()
        assign_space = FunctionSpace(mesh, "CG", 1)
        assigning_func = project(Constant(-70), assign_space)
        assigner = FunctionAssigner(W.sub(1), assign_space)
        assigner.assign(self.UV_.sub(1), assigning_func)

        assigner = FunctionAssigner(W.sub(1), assign_space)
        assigner.assign(self.UV.sub(1), assigning_func)

        _, foo = self.UV_.split(True)
        print(foo.vector().array())

        G = (v - V_)/dtc*w*dx
        G += inner(Mi*grad(v), grad(w))*dx
        G += inner(Mi*grad(u), grad(w))*dx
        G += inner(Mi*grad(v), grad(q))*dx
        G += inner((Mi + Me)*grad(u), grad(q))*dx
        # G += (lam*u + l*q)*dx
        G += stimulus*w*ds

        # G += ect_current*q*ds

        a, L = system(G)
        return a, L

    def _create_solver(self):
        self._lhs, self._rhs = self._create_variational_forms()

        self._lhs_matrix = assemble(self._lhs)
        # self._rhs_vector = Vector(self._mesh.mpi_comm(), self._lhs_matrix.size(0))

        self.linear_solver = PETScKrylovSolver("gmres")

        self.linear_solver.parameters.absolute_tolerance = None
        self.linear_solver.parameters.convergence_norm_type = None
        self.linear_solver.parameters.divergence_limit = None
        self.linear_solver.parameters.error_on_nonconvergence = None
        self.linear_solver.parameters.maximum_iterations = None
        self.linear_solver.parameters.monitor_convergence = True
        self.linear_solver.parameters.relative_tolerance = None

        self.linear_solver.parameters.nonzero_initial_guess = True
        self.linear_solver.parameters.report = False

        self.linear_solver.set_operator(self._lhs_matrix)

    def _step(self):
        # Solve problem

        b = assemble(self._rhs)

        b_copy = b.array().copy(True)
        avg_b = b_copy[::2]
        b.array()[::2] -= avg_b.sum()/avg_b.size

        # b -= b.sum()/b.size() # This is a bad idea

        self.linear_solver.solve(
            self.UV.vector(),
            b,
        )
        print("norm: ", b.norm("l2"))

    def solve(self, t0, T):
        self._create_solver()
        ufile = File("results/u.pvd")
        vfile = File("results/v.pvd")


        print("="*50)
        print(self.UV.vector().array()[::2])
        print(self.UV.vector().array()[1::2])
        print("="*50)

        t = t0
        while t <= T:
            t += dt
            self.time.assign(t)

            self._step()
            self.UV_.assign(self.UV) # update solution on previous time step with current solution 

            UE, V = self.UV.split(True)
            print(max(V.vector().array()))
            print(max(UE.vector().array()))


            ufile << UE
            vfile << V


if __name__ == "__main__":
    # mesh = Mesh("merge.xml.gz")
    mesh = UnitCubeMesh(10, 10, 10)
    dt = 0.001
    bdsolver = BDSolver(mesh, dt)
    bdsolver.solve(0, 1)
