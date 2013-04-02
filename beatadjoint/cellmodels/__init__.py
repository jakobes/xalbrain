# Base class for cardiac cell models
from cardiaccellmodel import CardiacCellModel

# Manually written specialized cell models
from nocellmodel import NoCellModel
from fitzhughnagumo_manual import FitzHughNagumoManual as OriginalFitzHughNagumo

# Automatically generated specialized cell models
from fitzhughnagumo import *
from tentusscher_2004_mcell import *

