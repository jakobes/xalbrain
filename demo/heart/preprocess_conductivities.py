from dolfin import *

# Do the serial stuff here first

parameters["reorder_dofs"] = False # Crucial!

def generate_conductivities():

    chi = 2000.0   # Membrane surface-to-volume ratio (1/cm), value from book
    C_m = 1.0      # Membrane capacitance per unit area (micro F/(cm^2))

    info("Loading mesh")
    mesh = Mesh("data/mesh115_refined.xml.gz")
    mesh.coordinates()[:] /= 1000.0 # Scale mesh from micrometer to millimeter
    mesh.coordinates()[:] /= 4.0    # Scale mesh as indicated by Johan

    # Load fibers and sheets
    info("Loading fibers and sheets")
    Vv = VectorFunctionSpace(mesh, "DG", 0)
    fiber = Function(Vv)
    File("data/fibers.xml.gz") >> fiber
    sheet = Function(Vv)
    File("data/sheet.xml.gz") >> sheet
    cross_sheet = Function(Vv)
    File("data/cross_sheet.xml.gz") >> cross_sheet

    # Load ischemic region (scalar function between 0-1, where 0 is ischemic)
    info("Loading ischemic region")
    V = FunctionSpace(mesh, "CG", 1)
    ischemic = Function(V)
    File("data/ischemic_region.xml.gz") >> ischemic
    ischemic_array = ischemic.vector().array()

    # Healthy and ischemic conductivities
    # (All values in mS/cm (milli-Siemens per centimeter)

    # Extracellular:
    g_el = 6.25/(C_m*chi) # Fiber
    g_et = 2.36/(C_m*chi) # Sheet
    g_et = 2.36/(C_m*chi) # Cross-sheet

    # Intracellular:
    g_il = 1.74/(C_m*chi)   # Fiber
    g_it = 0.192/(C_m*chi)  # Sheet
    g_it = 0.192/(C_m*chi)  # Cross-sheet

    # Extracellular:
    g_el_isch = 3.125/(C_m*chi) # Fiber
    g_et_isch = 1.18/(C_m*chi) # Sheet
    g_et_isch = 1.18/(C_m*chi) # Cross-sheet

    # Intracellular:
    g_il_isch = 0.125/(C_m*chi)  # Fiber
    g_it_isch = 0.125/(C_m*chi)  # Sheet
    g_it_isch = 0.125/(C_m*chi)  # Cross-sheet

    info("Creating conductivities")
    # Combine info into 2x2 distinct conductivity functions:
    g_el_field = Function(V)
    g_el_field.vector()[:] = (1-ischemic_array)*g_el_isch+ischemic_array*g_el
    g_et_field = Function(V)
    g_et_field.vector()[:] = (1-ischemic_array)*g_et_isch+ischemic_array*g_et
    g_il_field = Function(V)
    g_il_field.vector()[:] = (1-ischemic_array)*g_il_isch+ischemic_array*g_il
    g_it_field = Function(V)
    g_it_field.vector()[:] = (1-ischemic_array)*g_it_isch+ischemic_array*g_it

    # Store these
    info("Storing conductivities")
    file = File("data/g_el_field.xml.gz")
    file << g_el_field
    file = File("data/g_et_field.xml.gz")
    file << g_et_field
    file = File("data/g_il_field.xml.gz")
    file << g_il_field
    file = File("data/g_it_field.xml.gz")
    file << g_it_field



if __name__ == "__main__":

    generate_conductivities()
