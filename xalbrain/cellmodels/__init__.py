from .cardiaccellmodel import (
    CardiacCellModel,
    MultiCellModel,
)

from .nocellmodel import NoCellModel

from .adex_slow import AdexManual
from .logistic import LogisticCellModel
from .coupled import TestCellModel
from .adex import Adex

# Only add supported cell model here if it is tested to actually run
# with some multistage discretization
# TODO: Enforce this
SUPPORTED_CELL_MODELS = [
    NoCellModel,
    AdexManual,
    Adex,
]

__all__ = [
    "NoCellModel",
    "AdexManual",
    "Adex",
    "CardiacCellModel",
    "MultiCellModel",
    "Wei",
    "Cressman",
    "Cressman_Iext",
]

from .beeler_reuter_1977 import Beeler_reuter_1977
from .fitzhughnagumo_manual import FitzHughNagumoManual
from .fitzhughnagumo import Fitzhughnagumo
from .rogers_mcculloch_manual import RogersMcCulloch
from .tentusscher_2004_mcell import Tentusscher_2004_mcell
from .tentusscher_panfilov_2006_epi_cell import Tentusscher_panfilov_2006_epi_cell
from .fenton_karma_1998_BR_altered import Fenton_karma_1998_BR_altered
from .fenton_karma_1998_MLR1_altered import Fenton_karma_1998_MLR1_altered
from .grandi_pasqualini_bers_2010 import Grandi_pasqualini_bers_2010
from .adex_slow import AdexManual
from .adex import Adex
from .wei_manual import Wei
from .cressman_manual import Cressman
from .noble_manual import Noble
from .cressman_Iext_manual import Cressman_Iext

# Only add supported cell model here if it is tested to actually run
# with some multistage discretization
SUPPORTED_CELL_MODELS += [
    Beeler_reuter_1977,
    FitzHughNagumoManual,
    RogersMcCulloch,
    Tentusscher_2004_mcell,
    Tentusscher_panfilov_2006_epi_cell,
    Fenton_karma_1998_MLR1_altered,
    Fenton_karma_1998_BR_altered,
]

__all__ += [
    "Beeler_reuter_1977",
    "FitzHughNagumoManual",
    "Fitzhughnagumo",
    "RogersMcCulloch",
    "Tentusscher_2004_mcell",
    "Tentusscher_panfilov_2006_epi_cell",
    "Fenton_karma_1998_MLR1_altered",
    "Fenton_karma_1998_BR_altered",
]
