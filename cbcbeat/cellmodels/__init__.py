import glob
import os
import importlib
import types

all_names = set()

# Base class for cardiac cell models
import cardiaccellmodel
from cardiaccellmodel import CardiacCellModel

# Get absolut path to module
module_dir = os.sep.join(os.path.abspath(cardiaccellmodel.__file__).split(os.sep)[:-1])

# Iterate over modules and collect CardiacCellModels
supported_cell_models = set()
for module_path in glob.glob(os.path.join(module_dir, "*.py")):
    module_str = os.path.basename(module_path)[:-3]
    if module_str in ["__init__", "cardiaccellmodel"]:
        continue
    module = importlib.import_module("cbcbeat.cellmodels."+module_str)
    for name, attr in module.__dict__.items():
        if isinstance(attr, types.ClassType) and issubclass(attr, CardiacCellModel):
            supported_cell_models.add(attr)
            globals()[name] = attr
            all_names.add(name)

# Remove base class
supported_cell_models.remove(CardiacCellModel)
supported_cell_models = tuple(supported_cell_models)

# All CardiacCellModel names
__all__ = list(all_names)
__all__.append("supported_cell_models")
