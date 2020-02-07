"""
    pint.converters
    ~~~~~~~~~~~~~~~

    Functions and classes related to unit conversions.

    :copyright: 2016 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from .compat import log, exp, HAS_NUMPY


class Converter:
    """Base class for value converters."""

    is_multiplicative = True

    def to_reference(self, value, inplace=False):
        return value

    def from_reference(self, value, inplace=False):
        return value


class ScaleConverter(Converter):
    """A linear transformation

    Parameters
    ----------

    scale : float
        scaling factor for linear unit conversion

    inplace : bool
        controls if computation is done in place

    """

    is_multiplicative = True

    def __init__(self, scale):
        self.scale = scale

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


class OffsetConverter(Converter):
    """An affine transformation

    Parameters
    ----------

    scale : float
        multiplicative factor for unit conversion

    offset : float
        offset correction for unit conversion

    inplace : bool
        controls if computation is done in place

    """

    def __init__(self, scale, offset):
        self.scale = scale
        self.offset = offset

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


class LogarithmicConverter(Converter):
    """ Converts between linear units and logarithmic units, such as dB, octave, neper or pH.

    Q_log = logfactor * log( Q_lin / scale ) / log(log_base)

    Parameters
    ----------

    scale : float
        unit of reference at denominator for logarithmic unit conversion

    logbase : float
        base of logarithm used in the logarithmic unit conversion

    logfactor : float
        factor multupled to logarithm for unit conversion

    inplace : bool
        controls if computation is done in place

    """
    def __init__(self, scale, logbase, logfactor):
        """

        Parameters
        ----------

        scale : float
            unit of reference at denominator inside logarithm for unit conversion

        logbase: float
            base of logarithm used in unit conversion

        logfactor: float
            factor multiplied to logarithm for unit conversion
        """

        if HAS_NUMPY is False:
            print("'numpy' package is not installed. Will use math.log() "
                  "for logarithmic units.")

        self.scale = scale
        self.logbase = logbase
        self.logfactor = logfactor

    def to_reference(self, value, inplace=False):
        if inplace:
            value /= self.scale
            value = log(value)
            value *= self.logfactor / log(self.logbase)
        else:
            value = self.logfactor * log(value / self.scale) / log(self.logbase)

        return value

    def from_reference(self, value, inplace=False):
        if inplace:
            value /= self.logfactor
            value *= self.logbase
            value = exp(value)
            value *= self.scale
        else:
            value = self.scale * exp(log(self.logbase) * (value / self.logfactor))

        return value
