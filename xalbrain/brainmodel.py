import dolfin as df

from xalbrain.cellmodels import CardiacCellModel

from typing import (
    Dict,
    Optional,
    Union,
)

from coupled_utils import (
    CellTags,
    InterfaceTags,
)


class CoupledBrainModel:
    def __init__(
        self,
        *,
        time: df.Constant,
        mesh: df.Mesh,
        cell_model: CardiacCellModel,
        cell_function: df.MeshFunction,
        cell_tags: CellTags,
        interface_function: df.MeshFunction,
        interface_tags: InterfaceTags,
        intracellular_conductivity: Dict[int, df.Expression],
        other_conductivity: Dict[int, df.Expression],
        neumann_boundary_condition: Dict[int, df.Expression] = None,
        external_stimulus: Dict[int, df.Expression] = None,
        surface_to_volume_factor: Union[float, df.Constant] = None,
        membrane_capacitance: Union[float, df.Constant] = None
    ):
        """
        other conductivity is either the extracellular conductivity or the ratio of
        extracellular to intracellular conduictivity for bidomain and monodonain model
        respectively.

        This class is also a candidate for subclassing and add Neumann BC stuff
        """
        self._cell_model = cell_model
        self._time = time
        self._mesh = mesh
        self._cell_function = cell_function
        self._interface_function = interface_function

        self._intracellular_conductivity = intracellular_conductivity
        self._other_conductivity = other_conductivity

        self._cell_tags = cell_tags
        self._interface_tags = interface_tags
        self._neumann_boundary_condition = neumann_boundary_condition
        self._external_stimulus = external_stimulus

        if surface_to_volume_factor is None:
            self._surface_to_volume_factor = df.Constant(1)
        else:
            self._surface_to_volume_factor = df.Constant(surface_to_volume_factor)

        if membrane_capacitance is None:
            self._membrane_capacitance = df.Constant(1)
        else:
            self._membrane_capacitance = df.Constant(membrane_capacitance)

    @property
    def cell_model(self):
        return self._cell_model

    @property
    def time(self) -> df.Constant:
        return self._time

    @property
    def mesh(self) -> df.mesh:
        return self._mesh

    @property
    def cell_function(self) -> df.MeshFunction:
        return self._cell_function

    @property
    def interface_function(self) -> df.MeshFunction:
        return self._interface_function

    @property
    def intracellular_conductivity(self) -> Dict[int, df.Expression]:
        return self._intracellular_conductivity

    @property
    def extracellular_conductivity(self) -> Dict[int, df.Expression]:
        return self._other_conductivity

    @property
    def cell_tags(self) -> CellTags:
        return self._cell_tags

    @property
    def interface_tags(self) -> InterfaceTags:
        return self._interface_tags

    @property
    def neumann_boundary_condition(self) -> Optional[Dict[int, df.Expression]]:
        return self._neumann_boundary_condition

    @property
    def surface_to_volume_factor(self) -> df.Constant:
        return self._surface_to_volume_factor

    @property
    def membrane_capacitance(self) -> df.Constant:
        return self._membrane_capacitance
