import dolfin as df

from xalbrain.cellmodels import CardiacCellModel

from collections import OrderedDict

from typing import Dict


class MorrisLecar(CardiacCellModel):

    def __init__(self, params: df.parameters=None, init_conditions: Dict[str, float]=None) -> None:
        CardiacCellModel.__init__(self, params, init_conditions)

    @staticmethod
    def default_initial_conditions() -> OrderedDict:
        ic = OrderedDict([
            ("V", -60),
            ("N", 0)
        ])
        return ic
