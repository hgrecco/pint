"""
    pint.converters
    ~~~~~~~~~~~~~~~

    Functions and classes related to unit conversions.

    :copyright: 2016 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from dataclasses import dataclass
from dataclasses import fields as dc_fields

from .compat import HAS_NUMPY, exp, log  # noqa: F401


@dataclass(frozen=True)
class Converter:
    """Base class for value converters."""

    _subclasses = []
    _param_names_to_subclass = {}

    @property
    def is_multiplicative(self):
        return True

    @property
    def is_logarithmic(self):
        return False

    def to_reference(self, value, inplace=False):
        return value

    def from_reference(self, value, inplace=False):
        return value

    def __init_subclass__(cls, **kwargs):
        # Get constructor parameters
        super().__init_subclass__(**kwargs)
        cls._subclasses.append(cls)

    @classmethod
    def get_field_names(cls, new_cls):
        return frozenset((p.name for p in dc_fields(new_cls)))

    @classmethod
    def preprocess_kwargs(cls, **kwargs):
        return None

    @classmethod
    def from_arguments(cls, **kwargs):
        kwk = frozenset(kwargs.keys())
        try:
            new_cls = cls._param_names_to_subclass[kwk]
        except KeyError:
            for new_cls in cls._subclasses:
                p_names = frozenset((p.name for p in dc_fields(new_cls)))
                if p_names == kwk:
                    cls._param_names_to_subclass[kwk] = new_cls
                    break
            else:
                params = "(" + ", ".join(tuple(kwk)) + ")"
                raise ValueError(
                    f"There is no class registered for parameters {params}"
                )

        kw = new_cls.preprocess_kwargs(**kwargs)
        if kw is None:
            return new_cls(**kwargs)
        return cls.from_arguments(**kw)


@dataclass(frozen=True)
class ScaleConverter(Converter):
    """A linear transformation."""

    scale: float

    def to_reference(self, value, inplace=False):
        if inplace:
            value *= self.scale
        else:
            value = value * self.scale

        return value

    def from_reference(self, value, inplace=False):
        if inplace:
            value /= self.scale
        else:
            value = value / self.scale

        return value


@dataclass(frozen=True)
class OffsetConverter(ScaleConverter):
    """An affine transformation."""

    offset: float

    @property
    def is_multiplicative(self):
        return self.offset == 0

    def to_reference(self, value, inplace=False):
        if inplace:
            value *= self.scale
            value += self.offset
        else:
            value = value * self.scale + self.offset

        return value

    def from_reference(self, value, inplace=False):
        if inplace:
            value -= self.offset
            value /= self.scale
        else:
            value = (value - self.offset) / self.scale

        return value

    @classmethod
    def preprocess_kwargs(cls, **kwargs):
        if "offset" in kwargs and kwargs["offset"] == 0:
            return {"scale": kwargs["scale"]}
        return None


@dataclass(frozen=True)
class LogarithmicConverter(ScaleConverter):
    """Converts between linear units and logarithmic units, such as dB, octave, neper or pH.
    Q_log = logfactor * log( Q_lin / scale ) / log(log_base)

    Parameters
    ----------
    scale : float
        unit of reference at denominator for logarithmic unit conversion
    logbase : float
        base of logarithm used in the logarithmic unit conversion
    logfactor : float
        factor multiplied to logarithm for unit conversion
    inplace : bool
        controls if computation is done in place
    """

    logbase: float
    logfactor: float

    @property
    def is_multiplicative(self):
        return False

    @property
    def is_logarithmic(self):
        return True

    def from_reference(self, value, inplace=False):
        """Converts value from the reference unit to the logarithmic unit

        dBm   <------   mW
        y dBm = 10 log10( x / 1mW )
        """
        if inplace:
            value /= self.scale
            if HAS_NUMPY:
                log(value, value)
            else:
                value = log(value)
            value *= self.logfactor / log(self.logbase)
        else:
            value = self.logfactor * log(value / self.scale) / log(self.logbase)

        return value

    def to_reference(self, value, inplace=False):
        """Converts value to the reference unit from the logarithmic unit

        dBm   ------>   mW
        y dBm = 10 log10( x / 1mW )
        """
        if inplace:
            value /= self.logfactor
            value *= log(self.logbase)
            if HAS_NUMPY:
                exp(value, value)
            else:
                value = exp(value)
            value *= self.scale
        else:
            value = self.scale * exp(log(self.logbase) * (value / self.logfactor))

        return value
