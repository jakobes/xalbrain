import numpy
import pylab

def extract_data(filename):
    file = open(filename)
    lines = file.readlines()
    evolutions = [[float(f) for f in line.split(" ") if f]
                  for line in lines]
    e = numpy.array(evolutions)

    return e

def plot_data(e, times=None, show=True, ylabel="v", title="Title"):

    pylab.figure()
    for i in range(len(e[0, :])):
        if times is not None:
            pylab.plot(times, e[:, i], label="Point %d" % i)
        else:
            pylab.plot(e[:, i], label="Point %d" % i)

    pylab.legend()
    pylab.xlabel("t")
    pylab.ylabel("%s(x_i, t)" % ylabel)
    pylab.grid(True)
    pylab.ylim((-100, 50))
    pylab.title(title)
    if show:
        pylab.show()

def compare_data(e0, e1, times=None, show=True, ylabel="v", title="Title",
                 versions=("0", "1"), store=True):

    for i in range(len(e0[0, :])):
        pylab.figure()
        if times is not None:
            pylab.plot(times, e0[:, i], label="Point %d" % i)
            pylab.plot(times, e1[:, i], label="Point %d" % i)
        else:
            pylab.plot(e0[:, i], label="Point %d (%s)" % (i, versions[0]))
            pylab.plot(e1[:, i], label="Point %d (%s)" % (i, versions[1]))

        pylab.legend()
        pylab.xlabel("t")
        pylab.ylabel("%s(x_i, t)" % ylabel)
        pylab.grid(True)
        pylab.ylim((-100, 50))
        pylab.title(title)

        if store:
            pylab.savefig("pngs/comparison_%s_%d" % (ylabel, i))

    if show:
        pylab.show()


if __name__ == "__main__":

    for variable in ("v", "u"):
        e0 = extract_data("pycc-results/%s.txt" % variable)
        e1 = extract_data("results-direct/%s.txt" % variable)
        compare_data(e0, e1, ylabel=variable,
                     show=True, versions=("pycc", "fenics"),
                     title="Comparing %s data for pycc and fenics" % variable)

    #filename = "pycc-results/u.txt"
    #e = extract_data(filename)
    #plot_data(e, ylabel="u", title="pycc-results")
