"""
    pint.definitions
    ~~~~~~~~~~~~~~~~

    Kept for backwards compatibility

    :copyright: 2022 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from . import errors
from ._vendor import flexparser as fp
from .delegates import ParserConfig, txt_defparser


class Definition:
    """This is kept for backwards compatibility"""

    @classmethod
    def from_string(cls, s: str, non_int_type=float):
        cfg = ParserConfig(non_int_type)
        parser = txt_defparser.DefParser(cfg, None)
        pp = parser.parse_string(s)
        for definition in parser.iter_parsed_project(pp):
            if isinstance(definition, Exception):
                raise errors.DefinitionSyntaxError(str(definition))
            if not isinstance(definition, (fp.BOS, fp.BOF, fp.BOS)):
                return definition
