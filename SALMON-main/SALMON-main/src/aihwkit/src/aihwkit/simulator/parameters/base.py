# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Base classes for the RPUConfig."""

from typing import ClassVar, Type, Union, Any, TYPE_CHECKING
from dataclasses import dataclass, field

from aihwkit.simulator.parameters.utils import MappingParameter, PrePostProcessingParameter
from aihwkit.simulator.parameters.helpers import _PrintableMixin, tile_parameters_to_bindings
from aihwkit.simulator.rpu_base import devices


if TYPE_CHECKING:
    from aihwkit.simulator.configs.configs import (
        InferenceRPUConfig,
        SingleRPUConfig,
        UnitCellRPUConfig,
        TorchInferenceRPUConfig,
        DigitalRankUpdateRPUConfig,
    )
    from aihwkit.simulator.tiles.base import BaseTile


RPUConfigGeneric = Union[
    "InferenceRPUConfig",
    "SingleRPUConfig",
    "UnitCellRPUConfig",
    "TorchInferenceRPUConfig",
    "DigitalRankUpdateRPUConfig",
]


@dataclass
class RPUConfigBase:
    """Base class of all RPUConfigs."""

    tile_class: ClassVar[Type]
    """Tile class that correspond to the RPUConfig. Needs to be
    defined in the derived class."""

    def as_bindings(self) -> devices.AnalogTileParameter:
        """Return a representation of this instance as a simulator bindings object."""
        return tile_parameters_to_bindings(self)

    def compatible_with(self, tile_class_name: str) -> bool:
        """Tests whether the RPUConfig is compatile with a given ``TileModule`` class.

        Args:
            tile_class_name: name of the TileModule class

        Returns:

            Whehter the class is compatible. By default only the
            class that is defined in the ``tile_class`` property of
            the RPUConfig
        """
        return tile_class_name == self.tile_class.__name__

    def get_default_tile_module_class(self, out_size: int = 0, in_size: int = 0) -> Type:
        """Returns the default TileModule class."""
        # pylint: disable=unused-argument
        return self.tile_class

    def create_tile(self, *args: Any, **kwargs: Any) -> "BaseTile":
        """Created a tile with this configuration.

        Short-cut for instantiating ``self.tile_class`` with given parameters.
        """
        return self.tile_class(*args, rpu_config=self, **kwargs)


@dataclass
class MappableRPU(RPUConfigBase, _PrintableMixin):
    """Defines the mapping parameters and utility factories"""

    tile_array_class: ClassVar[Type]
    """Tile array class that correspond to the RPUConfig.

    This is used to build logical arrays of tiles. Needs to be defined
    in the derived class.
    """

    mapping: MappingParameter = field(default_factory=MappingParameter)
    """Parameter related to mapping weights to tiles for supporting modules."""

    def get_default_tile_module_class(self, out_size: int = 0, in_size: int = 0) -> Type:
        """Returns the default TileModule class."""
        if self.mapping.max_input_size == 0 and self.mapping.max_output_size == 0:
            return self.tile_class
        if self.mapping.max_input_size < in_size or self.mapping.max_output_size < out_size:
            return self.tile_array_class
        return self.tile_class


@dataclass
class PrePostProcessingRPU(RPUConfigBase, _PrintableMixin):
    """Defines the pre-post parameters and utility factories"""

    pre_post: PrePostProcessingParameter = field(default_factory=PrePostProcessingParameter)
    """Parameter related digital pre and post processing."""
