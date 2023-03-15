"""
    flexcache.flexcache
    ~~~~~~~~~~~~~~~~~~~

    Classes for persistent caching and invalidating cached objects,
    which are built from a source object and a (potentially expensive)
    conversion function.

    Header
    ------
    Contains summary information about the source object that will
    be saved together with the cached file.

    It's capabilities are divided in three groups:
    - The Header itself which contains the information that will
      be saved alongside the cached file
    - The Naming logic which indicates how the cached filename is
      built.
    - The Invalidation logic which indicates whether a cached file
      is valid (i.e. truthful to the actual source file).

    DiskCache
    ---------
    Saves and loads to the cache a transformed versions of a source object.

    :copyright: 2022 by flexcache Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

import abc
import hashlib
import json
import pathlib
import pickle
import platform
import typing
from dataclasses import asdict as dc_asdict
from dataclasses import dataclass
from dataclasses import fields as dc_fields
from typing import Any, Iterable

#########
# Header
#########


@dataclass(frozen=True)
class BaseHeader(abc.ABC):
    """Header with no information except the converter_id

    All header files must inherit from this.
    """

    # The actual source of the data (or a reference to it)
    # that is going to be converted.
    source: Any

    # An identification of the function that is used to
    # convert the source into the result object.
    converter_id: str

    _source_type = object

    def __post_init__(self):
        # TODO: In more modern python versions it would be
        # good to check for things like tuple[str].
        if not isinstance(self.source, self._source_type):
            raise TypeError(
                f"Source must be {self._source_type}, " f"not {type(self.source)}"
            )

    def for_cache_name(self) -> typing.Generator[bytes]:
        """The basename for the cache file is a hash hexdigest
        built by feeding this collection of values.

        A class can provide it's own set of values by rewriting
        `_for_cache_name`.
        """
        for el in self._for_cache_name():
            if isinstance(el, str):
                yield el.encode("utf-8")
            else:
                yield el

    def _for_cache_name(self) -> typing.Generator[bytes | str]:
        """The basename for the cache file is a hash hexdigest
        built by feeding this collection of values.

        Change the behavior by writing your own.
        """
        yield self.converter_id

    @abc.abstractmethod
    def is_valid(self, cache_path: pathlib.Path) -> bool:
        """Return True if the cache_path is an cached version
        of the source_object represented by this header.
        """


@dataclass(frozen=True)
class BasicPythonHeader(BaseHeader):
    """Header with basic Python information."""

    system: str = platform.system()
    python_implementation: str = platform.python_implementation()
    python_version: str = platform.python_version()


#####################
# Invalidation logic
#####################


class InvalidateByExist:
    """The cached file is valid if exists and is newer than the source file."""

    def is_valid(self, cache_path: pathlib.Path) -> bool:
        return cache_path.exists()


class InvalidateByPathMTime(abc.ABC):
    """The cached file is valid if exists and is newer than the source file."""

    @property
    @abc.abstractmethod
    def source_path(self) -> pathlib.Path:
        ...

    def is_valid(self, cache_path: pathlib.Path):
        return (
            cache_path.exists()
            and cache_path.stat().st_mtime > self.source_path.stat().st_mtime
        )


class InvalidateByMultiPathsMtime(abc.ABC):
    """The cached file is valid if exists and is newer than the newest source file."""

    @property
    @abc.abstractmethod
    def source_paths(self) -> pathlib.Path:
        ...

    @property
    def newest_date(self):
        return max((t.stat().st_mtime for t in self.source_paths), default=0)

    def is_valid(self, cache_path: pathlib.Path):
        return cache_path.exists() and cache_path.stat().st_mtime > self.newest_date


###############
# Naming logic
###############


class NameByFields:
    """Name is built taking into account all fields in the Header
    (except the source itself).
    """

    def _for_cache_name(self):
        yield from super()._for_cache_name()
        for field in dc_fields(self):
            if field.name not in ("source", "converter_id"):
                yield getattr(self, field.name)


class NameByFileContent:
    """Given a file source object, the name is built from its content."""

    _source_type = pathlib.Path

    @property
    def source_path(self) -> pathlib.Path:
        return self.source

    def _for_cache_name(self):
        yield from super()._for_cache_name()
        yield self.source_path.read_bytes()

    @classmethod
    def from_string(cls, s: str, converter_id: str):
        return cls(pathlib.Path(s), converter_id)


@dataclass(frozen=True)
class NameByObj:
    """Given a pickable source object, the name is built from its content."""

    pickle_protocol: int = pickle.HIGHEST_PROTOCOL

    def _for_cache_name(self):
        yield from super()._for_cache_name()
        yield pickle.dumps(self.source, protocol=self.pickle_protocol)


class NameByPath:
    """Given a file source object, the name is built from its resolved path."""

    _source_type = pathlib.Path

    @property
    def source_path(self) -> pathlib.Path:
        return self.source

    def _for_cache_name(self):
        yield from super()._for_cache_name()
        yield bytes(self.source_path.resolve())

    @classmethod
    def from_string(cls, s: str, converter_id: str):
        return cls(pathlib.Path(s), converter_id)


class NameByMultiPaths:
    """Given multiple file source object, the name is built from their resolved path
    in ascending order.
    """

    _source_type = tuple

    @property
    def source_paths(self) -> tuple[pathlib.Path]:
        return self.source

    def _for_cache_name(self):
        yield from super()._for_cache_name()
        yield from sorted(bytes(p.resolve()) for p in self.source_paths)

    @classmethod
    def from_strings(cls, ss: Iterable[str], converter_id: str):
        return cls(tuple(pathlib.Path(s) for s in ss), converter_id)


class NameByHashIter:
    """Given multiple hashes, the name is built from them in ascending order."""

    _source_type = tuple

    def _for_cache_name(self):
        yield from super()._for_cache_name()
        yield from sorted(h for h in self.source)


class DiskCache:
    """A class to store and load cached objects to disk, which
    are built from a source object and conversion function.

    The basename for the cache file is a hash hexdigest
    built by feeding a collection of values determined by
    the Header object.

    Parameters
    ----------
    cache_folder
        indicates where the cache files will be saved.
    """

    # Maps classes to header class
    _header_classes: dict[type, BaseHeader] = None

    # Hasher object constructor (e.g. a member of hashlib)
    # must implement update(b: bytes) and hexdigest() methods
    _hasher = hashlib.sha1

    # If True, for each cached file the header is also stored.
    _store_header: bool = True

    def __init__(self, cache_folder: str | pathlib.Path):
        self.cache_folder = pathlib.Path(cache_folder)
        self.cache_folder.mkdir(parents=True, exist_ok=True)
        self._header_classes = self._header_classes or {}

    def register_header_class(self, object_class: type, header_class: BaseHeader):
        self._header_classes[object_class] = header_class

    def cache_stem_for(self, header: BaseHeader) -> str:
        """Generate a hash representing the basename of a memoized file
        for a given header.

        The naming strategy is defined by the header class used.
        """
        hd = self._hasher()
        for value in header.for_cache_name():
            hd.update(value)
        return hd.hexdigest()

    def cache_path_for(self, header: BaseHeader) -> pathlib.Path:
        """Generate a Path representing the location of a memoized file
        for a given filepath or object.

        The naming strategy is defined by the header class used.
        """
        h = self.cache_stem_for(header)
        return self.cache_folder.joinpath(h).with_suffix(".pickle")

    def _get_header_class(self, source_object) -> BaseHeader:
        for k, v in self._header_classes.items():
            if isinstance(source_object, k):
                return v
        raise TypeError(f"Cannot find header class for {type(source_object)}")

    def load(self, source_object, converter=None, pass_hash=False) -> tuple[Any, str]:
        """Given a source_object, return the converted value stored
        in the cache together with the cached path stem

        When the cache is not found:
        - If a converter callable is given, use it on the source
          object, store the result in the cache and return it.
        - Return None, otherwise.

        Two signatures for the converter are valid:
        - source_object -> transformed object
        - (source_object, cached_path_stem) -> transformed_object

        To use the second one, use `pass_hash=True`.

        If you want to do the conversion yourself outside this class,
        use the converter argument to provide a name for it. This is
        important as the cached_path_stem depends on the converter name.
        """
        header_class = self._get_header_class(source_object)

        if isinstance(converter, str):
            converter_id = converter
            converter = None
        else:
            converter_id = getattr(converter, "__name__", "")

        header = header_class(source_object, converter_id)

        cache_path = self.cache_path_for(header)

        converted_object = self.rawload(header, cache_path)

        if converted_object:
            return converted_object, cache_path.stem
        if converter is None:
            return None, cache_path.stem

        if pass_hash:
            converted_object = converter(source_object, cache_path.stem)
        else:
            converted_object = converter(source_object)

        self.rawsave(header, converted_object, cache_path)

        return converted_object, cache_path.stem

    def save(self, converted_object, source_object, converter_id="") -> str:
        """Given a converted_object and its corresponding source_object,
        store it in the cache and return the cached_path_stem.
        """

        header_class = self._get_header_class(source_object)
        header = header_class(source_object, converter_id)
        return self.rawsave(header, converted_object, self.cache_path_for(header)).stem

    def rawload(
        self, header: BaseHeader, cache_path: pathlib.Path = None
    ) -> Any | None:
        """Load the converted_object from the cache if it is valid.

        The invalidating strategy is defined by the header class used.

        The cache_path is optional, it will be calculated from the header
        if not given.
        """
        if cache_path is None:
            cache_path = self.cache_path_for(header)

        if header.is_valid(cache_path):
            with cache_path.open(mode="rb") as fi:
                return pickle.load(fi)

    def rawsave(
        self, header: BaseHeader, converted, cache_path: pathlib.Path = None
    ) -> pathlib.Path:
        """Save the converted object (in pickle format) and
        its header (in json format) to the cache folder.

        The cache_path is optional, it will be calculated from the header
        if not given.
        """
        if cache_path is None:
            cache_path = self.cache_path_for(header)

        if self._store_header:
            with cache_path.with_suffix(".json").open("w", encoding="utf-8") as fo:
                json.dump({k: str(v) for k, v in dc_asdict(header).items()}, fo)
        with cache_path.open(mode="wb") as fo:
            pickle.dump(converted, fo)
        return cache_path


class DiskCacheByHash(DiskCache):
    """Convenience class used for caching conversions that take a path,
    naming by hashing its content.
    """

    @dataclass(frozen=True)
    class Header(NameByFileContent, InvalidateByExist, BaseHeader):
        pass

    _header_classes = {
        pathlib.Path: Header,
        str: Header.from_string,
    }


class DiskCacheByMTime(DiskCache):
    """Convenience class used for caching conversions that take a path,
    naming by hashing its full path and invalidating by the file
    modification time.
    """

    @dataclass(frozen=True)
    class Header(NameByPath, InvalidateByPathMTime, BaseHeader):
        pass

    _header_classes = {
        pathlib.Path: Header,
        str: Header.from_string,
    }
