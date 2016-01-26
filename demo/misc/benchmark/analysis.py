"""
This script produces the activation time plots for the Niederer et al 2011 benchmark
Phil.Trans. R. Soc. A 369,
"""
__author__ = "Simon W. Funke (simon@simula.no) and Marie Rognes, 2014"
__all__ = []

from fenics import *
from matplotlib import pyplot 
import numpy
import argparse


def print_times_p1p8(activation_time):
    # Evaluate the activation times at P1...P8
    evaluation_points = [(0, 0, 0), (0, 7, 0), (20, 0, 0), (20, 7, 0), 
                         (0, 0, 3), (0, 7, 3), (20, 0, 3), (20, 7, 3), 
                         (10, 3.5, 1.5)]
    for i, p in enumerate(evaluation_points):
        print "Activation time for point P%i: %f" % (i+1, activation_time(p))

def plot_p1p8_line(activation_time, Lx, Ly, Lz):
    # Plot along the P1 - P8 line
    res = 101
    x_coords = numpy.linspace(0, Lx, res)
    y_coords = numpy.linspace(0, Ly, res)
    z_coords = numpy.linspace(0, Lz, res)
    coords = zip(x_coords, y_coords, z_coords)

    dist = [numpy.linalg.norm(c) for c in coords]
    time = [activation_time(p) for p in coords]

    pyplot.plot(dist, time)
    pyplot.xlabel('Distance')
    pyplot.ylabel('Activation time (ms)')
    pyplot.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Analyse Niederer benchmark activation files.')
    parser.add_argument('filename', type=str, help='xml activation time file')

    args = parser.parse_args()
    mesh = Mesh(args.filename[:-4] + "_mesh.xml")
    V = FunctionSpace(mesh, "CG", 1)
    activation_time = Function(V, args.filename)

    # Define geometry parameters
    Lx = 20. # mm
    Ly = 7.  # mm
    Lz = 3.  # mm

    print_times_p1p8(activation_time)
    plot_p1p8_line(activation_time, Lx, Ly, Lz)
    plot(activation_time, title="Activation time") 
    interactive()
