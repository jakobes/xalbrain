from .cardiaccellmodel import (
    CardiacCellModel,
    MultiCellModel,
)

from .nocellmodel import NoCellModel

from .adex_slow import AdexManual
from .adex import Adex

# Only add supported cell model here if it is tested to actually run
# with some multistage discretization
# TODO: Enforce this 
supported_cell_models = [
    NoCellModel,
    AdexManual,
    Adex,
]

try:
    from .beeler_reuter_1977 import Beeler_reuter_1977
    from .fitzhughnagumo_manual import FitzHughNagumoManual
    from .rogers_mcculloch_manual import RogersMcCulloch
    from .tentusscher_2004_mcell import Tentusscher_2004_mcell
    from .tentusscher_panfilov_2006_epi_cell import Tentusscher_panfilov_2006_epi_cell
    from .fenton_karma_1998_BR_altered import Fenton_karma_1998_BR_altered
    from .fenton_karma_1998_MLR1_altered import Fenton_karma_1998_MLR1_altered
    from .grandi_pasqualini_bers_2010 import Grandi_pasqualini_bers_2010
    from .adex_slow import AdexManual
    from .adex import Adex

    # Only add supported cell model here if it is tested to actually run
    # with some multistage discretization
    supported_cell_models += [
        Beeler_reuter_1977,
        FitzHughNagumoManual,
        RogersMcCulloch,
        Tentusscher_2004_mcell,
        Tentusscher_panfilov_2006_epi_cell,
        Fenton_karma_1998_MLR1_altered,
        Fenton_karma_1998_BR_altered,
    ]
except:
    pass

# Convert to string
__all__ = list(map(lambda x: x.__name__, supported_cell_models))
