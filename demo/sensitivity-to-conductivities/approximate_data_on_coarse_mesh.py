
from dolfin import *

def main():
    parameters["allow_extrapolation"] = True

    mesh = Mesh("data/mesh115_refined.xml.gz")
    mesh.coordinates()[:] /= 1000.0 # Scale mesh from micrometer to millimeter
    mesh.coordinates()[:] /= 10.0   # Scale mesh from millimeter to centimeter
    mesh.coordinates()[:] /= 4.0    # Scale mesh as indicated by Johan/Molly

    # Load fibers and sheets
    info("Loading fibers and sheets")
    Vv = VectorFunctionSpace(mesh, "DG", 0)
    fiber = Function(Vv)
    File("data/fibers.xml.gz") >> fiber
    sheet = Function(Vv)
    File("data/sheet.xml.gz") >> sheet
    cross_sheet = Function(Vv)
    File("data/cross_sheet.xml.gz") >> cross_sheet

    info("Loading coarse mesh")
    coarse_mesh = Mesh("data/mesh115_coarse.xml.gz")
    Vv_c = VectorFunctionSpace(coarse_mesh, "DG", 0)

    info("Projecting data onto coarse mesh")
    fiber_c = project(fiber, Vv_c)
    sheet_c = project(sheet, Vv_c)
    cross_sheet_c = project(cross_sheet, Vv_c)

    plot(fiber_c)
    plot(sheet_c)
    plot(cross_sheet_c)

    info("Storing data")
    File("data/fibers_coarse.xml.gz") << fiber_c
    File("data/sheet_coarse.xml.gz") << sheet_c
    File("data/cross_sheet_coarse.xml.gz") << cross_sheet_c

    info("Success.")
    interactive()

if __name__ == "__main__":
    main()
