from __future__ import annotations

import pathlib
import typing as ty

import flexcache as fc
import flexparser as fp

from ...facets.plain import definitions
from ..base_defparser import ParserConfig

# from . import block, common, context, defaults, group, plain, system
from . import common, plain, quantitykind, unit

# class QudtRootBlock(
#     fp.RootBlock[
#         ty.Union[
#             plain.CommentDefinition,
#             # common.ImportDefinition,
#             # context.ContextDefinition,
#             # defaults.DefaultsDefinition,
#             # system.SystemDefinition,
#             # group.GroupDefinition,
#             # plain.AliasDefinition,
#             # plain.DerivedDimensionDefinition,
#             # plain.DimensionDefinition,
#             # plain.PrefixDefinition,
#             plain.UnitDefinition,
#         ],
#         ParserConfig,
#     ]
# ):
#     pass


class QudtRootBlock(
    fp.RootBlock[
        ty.Union[
            common.ImportDefinition,
            common.HeaderBlock,
            unit.UnitDefinitionBlock,
            quantitykind.QuantitykindDefinitionBlock,
            common.VoagDefinitionBlock,
            common.VaemDefinitionBlock,
            common.HttpDefinitionBlock,
        ],
        ParserConfig,
    ]
):
    pass


class _QudtParser(fp.Parser[QudtRootBlock, ParserConfig]):
    """Parser for the Qudt definition file, with cache."""

    _delimiters = {
        "#": (
            fp.DelimiterInclude.SPLIT_BEFORE,
            fp.DelimiterAction.CAPTURE_NEXT_TIL_EOL,
        ),
        **fp.SPLIT_EOL,
    }
    _root_block_class = QudtRootBlock
    _strip_spaces = True

    _diskcache: fc.DiskCache | None

    def __init__(self, config: ParserConfig, *args: ty.Any, **kwargs: ty.Any):
        self._diskcache = kwargs.pop("diskcache", None)
        super().__init__(config, *args, **kwargs)

    def parse_file(
        self, path: pathlib.Path
    ) -> fp.ParsedSource[QudtRootBlock, ParserConfig]:
        if self._diskcache is None:
            return super().parse_file(path)
        content, _basename = self._diskcache.load(path, super().parse_file)
        return content


class QudtParser:
    skip_classes: tuple[type, ...] = (
        fp.BOF,
        fp.BOR,
        fp.BOS,
        fp.EOS,
        plain.CommentDefinition,
        common.HeaderBlock,
        common.VoagDefinitionBlock,
        common.VaemDefinitionBlock,
        common.HttpDefinitionBlock,
    )

    def __init__(self, default_config: ParserConfig, diskcache: fc.DiskCache):
        self._default_config = default_config
        self._diskcache = diskcache

    def iter_parsed_project(
        self, parsed_project: fp.ParsedProject[QudtRootBlock, ParserConfig]
    ) -> ty.Generator[fp.ParsedStatement[ParserConfig], None, None]:
        last_location = None

        for dim in common.DIMENSIONS:
            yield definitions.DimensionDefinition("[" + dim + "]")
        for stmt in parsed_project.iter_blocks():
            print(104, stmt)
            if isinstance(stmt, fp.BOS):
                if isinstance(stmt, fp.BOF):
                    last_location = str(stmt.path)
                    continue
                elif isinstance(stmt, fp.BOR):
                    last_location = (
                        f"[package: {stmt.package}, resource: {stmt.resource_name}]"
                    )
                    continue
                else:
                    last_location = "orphan string"
                    continue

            if isinstance(stmt, self.skip_classes):
                continue
            if isinstance(
                stmt,
                (unit.UnitDefinitionBlock, quantitykind.QuantitykindDefinitionBlock),
            ):
                print(121)
                yield stmt.derive_definition()
                continue
            print(123)
            assert isinstance(last_location, str)
            if isinstance(stmt, common.DefinitionSyntaxError):
                stmt.set_location(last_location)
                raise stmt
            # elif isinstance(stmt, block.DirectiveBlock):
            #     for exc in stmt.errors:
            #         exc = common.DefinitionSyntaxError(str(exc))
            #         exc.set_position(*stmt.get_position())
            #         exc.set_raw(
            #             (stmt.opening.raw or "") + " [...] " + (stmt.closing.raw or "")
            #         )
            #         exc.set_location(last_location)
            #         raise exc

            # try:
            #     yield stmt.derive_definition()
            # except Exception as exc:
            #     exc = common.DefinitionSyntaxError(str(exc))
            #     exc.set_position(*stmt.get_position())
            #     exc.set_raw(stmt.opening.raw + " [...] " + stmt.closing.raw)
            #     exc.set_location(last_location)
            #     raise exc
            else:
                yield stmt

    def parse_file(
        self, filename: pathlib.Path | str, cfg: ParserConfig | None = None
    ) -> fp.ParsedProject[QudtRootBlock, ParserConfig]:
        return fp.parse(
            filename,
            _QudtParser,
            # cfg or self._default_config,
            # diskcache=self._diskcache,
            # strip_spaces=True,
            # delimiters=_QudtParser._delimiters,
        )

    def parse_string(
        self, content: str, cfg: ParserConfig | None = None
    ) -> fp.ParsedProject[QudtRootBlock, ParserConfig]:
        return fp.parse_bytes(
            content.encode("utf-8"),
            _QudtParser,
            cfg or self._default_config,
            diskcache=self._diskcache,
            strip_spaces=True,
            delimiters=_QudtParser._delimiters,
        )
