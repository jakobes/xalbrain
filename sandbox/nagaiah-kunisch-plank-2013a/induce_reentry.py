from beatadjoint import *

def main():

    n = 128
    mesh = Rectangle(0.0, 0.0, 2.0, 2.0, n, n)
    time = Constant(0.0)

    M_i = diag(as_vector([2.0e-3, 3.1e-4])) # S/cm
    M_e = diag(as_vector([2.0e-3, 1.3e-3]))

    cell_model =

    s1s2 = Expression("1.0")
    tissue = CardiacModel(mesh, time, M_i, M_e, cell_model, stimulus={0: s1s2})



if __name__ == "__main__":
    main()
