"""
    pint.diskmemo
    ~~~~~~~~~~~~~

    Functions for disk memoization.

    Files are stored in the CACHE FOLDER under a name generated
    by hashing with sha1 the raw filesystem path as encoded by
    fs.encode().

    :copyright: 2022 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

import hashlib
import pathlib
import pickle

# TODO: provide a correct cache dir, maybe using appdirs
CACHE_FOLDER = pathlib.Path().cwd()
CACHE_FOLDER = pathlib.Path("/Users/grecco/Documents/code/pint")


def cachepath_for(obj) -> pathlib.Path:
    """Generate a Path representing the location of a memoized file
    for a given filepath or object.
    """
    if isinstance(obj, pathlib.Path):
        hd = hashlib.sha1(bytes(obj.resolve()))
    else:
        hd = hashlib.sha1(pickle.dumps(obj))

    return CACHE_FOLDER.joinpath(hd.hexdigest()).with_suffix(".pickle")


def load(original: pathlib.Path, ignore_date=False):
    """Load and return the cached file if exists and it was created
    after the original file.

    If ignore_date is True, the file is loaded even if the cache
    is older thant the actual file.
    """

    if ignore_date:
        newest_date = 0
    else:
        newest_date = original.stat().st_mtime
    return rawload(cachepath_for(original), newest_date)


def save(obj, original: pathlib.Path) -> pathlib.Path:
    """Save the object (in pickle format) to the cache folder
    using a unique name generated using `cachepath_for`
    """
    return rawsave(obj, cachepath_for(original))


def load_referred(d: dict, ignore_date=False):
    """Load and return the cached file if exists and it was created
    after the original referring data structure.

    The datastructure is summarized in a dict that containes the
    modification time of different subelements as their values.

    If ignore_date is True, the file is loaded even if the cache
    is older thant the actual file.
    """
    keys = sorted(d.keys())
    if ignore_date or not keys:
        newest_date = 0
    else:
        newest_date = max(
            (mtime for mtime in d.values() if mtime is not None), default=0
        )
    return rawload(cachepath_for(keys), newest_date)


def save_referred(obj, d: dict):
    """Save the object (in pickle format) to the cache folder
    using a unique name generated using `cachepath_for`
    """

    keys = sorted(d.keys())
    return rawsave(obj, cachepath_for(keys))


def rawload(cachepath: pathlib.Path, newest_mtime):
    if not cachepath.exists():
        return
    if cachepath.stat().st_mtime > newest_mtime:
        with cachepath.open(mode="rb") as fi:
            return pickle.load(fi)
    return None


def rawsave(obj, cachepath: pathlib.Path) -> pathlib.Path:
    """Save the object (in pickle format) to the cache folder
    using a unique name generated using `cachepath_for`
    """
    with cachepath.open(mode="wb") as fo:
        pickle.dump(obj, fo)
    return cachepath
