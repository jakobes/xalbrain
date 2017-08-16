import glob
import os
import importlib
import types

# Base class for cardiac cell models
#import cardiaccellmodel
from .cardiaccellmodel import CardiacCellModel, MultiCellModel

from .beeler_reuter_1977 import Beeler_reuter_1977
from .fitzhughnagumo_manual import FitzHughNagumoManual
from .nocellmodel import NoCellModel
from .rogers_mcculloch_manual import RogersMcCulloch
from .tentusscher_2004_mcell import Tentusscher_2004_mcell
from .tentusscher_panfilov_2006_epi_cell import Tentusscher_panfilov_2006_epi_cell
from .fenton_karma_1998_BR_altered import Fenton_karma_1998_BR_altered
from .fenton_karma_1998_MLR1_altered import Fenton_karma_1998_MLR1_altered
from .grandi_pasqualini_bers_2010 import Grandi_pasqualini_bers_2010
from .adex import AdExManual

from .test_adex import Test_adex

# Only add supported cell model here if it is tested to actually run
# with some multistage discretization

supported_cell_models = (FitzHughNagumoManual,
                         NoCellModel,
                         RogersMcCulloch,
                         Beeler_reuter_1977,
                         Tentusscher_2004_mcell,
                         Tentusscher_panfilov_2006_epi_cell,
                         Fenton_karma_1998_MLR1_altered,
                         Fenton_karma_1998_BR_altered)#,
                         #Grandi_pasqualini_bers_2010)

# Iterate over modules and collect CardiacCellModels
#supported_cell_models = set()
#all_names = set()

# Get absolut path to module
#module_dir = os.sep.join(os.path.abspath(cardiaccellmodel.__file__).split(os.sep)[:-1])
# for module_path in glob.glob(os.path.join(module_dir, "*.py")):
#     module_str = os.path.basename(module_path)[:-3]
#     if module_str in ["__init__", "cardiaccellmodel"]:
#         continue
#     module = importlib.import_module("cbcbeat.cellmodels."+module_str)
#     for name, attr in module.__dict__.items():
#         if isinstance(attr, types.ClassType) and issubclass(attr, CardiacCellModel):
#             supported_cell_models.add(attr)
#             globals()[name] = attr
#             all_names.add(name)

# # Remove base class
# supported_cell_models.remove(CardiacCellModel)
# supported_cell_models = tuple(supported_cell_models)
# All CardiacCellModel names
#__all__ = [str(s) for s in supported_cell_models]
#__all__.append("supported_cell_models")
