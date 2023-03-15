"""
    pint.facets.measurement.registry
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    :copyright: 2022 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""


from __future__ import annotations

from ...compat import ufloat
from ...util import build_dependent_class, create_class_with_registry
from ..plain import PlainRegistry
from .objects import Measurement, MeasurementQuantity


class MeasurementRegistry(PlainRegistry):

    _quantity_class = MeasurementQuantity
    _measurement_class = Measurement

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__()

        cls.Measurement = build_dependent_class(
            cls, "Measurement", "_measurement_class"
        )

    def _init_dynamic_classes(self) -> None:
        """Generate subclasses on the fly and attach them to self"""
        super()._init_dynamic_classes()

        if ufloat is not None:
            self.Measurement = create_class_with_registry(self, self.Measurement)
        else:

            def no_uncertainties(*args, **kwargs):
                raise RuntimeError(
                    "Pint requires the 'uncertainties' package to create a Measurement object."
                )

            self.Measurement = no_uncertainties
