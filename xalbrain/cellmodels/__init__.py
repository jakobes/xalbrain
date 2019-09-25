from .cellmodel import (
    CellModel,
    MultiCellModel,
)

from .nocellmodel import NoCellModel

from .adex_slow import AdexManual
from .logistic import LogisticCellModel
from .coupled import TestCoupledCellModel
from .adex import Adex
from .wei_manual import Wei
from .cressman_manual import Cressman
from .fitzhughnagumo import FitzhughNagumo
from .fitzhughnagumo_manual import FitzHughNagumoManual


SUPPORTED_CELL_MODELS = [
    NoCellModel,
    AdexManual,
    Adex,
    LogisticCellModel,
    TestCoupledCellModel,
    Wei,
    Cressman,
    FitzhughNagumo,
    FitzHughNagumoManual,
]

__all__ = [
    "NoCellModel",
    "AdexManual",
    "Adex",
    "LogisticCellModel",
    "MultiCellModel",
    "Wei",
    "Cressman",
    "FitzhughNagumo",
    "FitzHughNagumoManual",
]
