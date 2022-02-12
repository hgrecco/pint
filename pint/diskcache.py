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


def cachepath_for(filepath: pathlib.Path) -> pathlib.Path:
    """Generate a Path representing the location of a memoized file
    for a given filepath.
    """
    hd = hashlib.sha1(bytes(filepath.resolve())).hexdigest()
    return CACHE_FOLDER.joinpath(hd).with_suffix(".pickle")


def load(original: pathlib.Path, ignore_date=False):
    """Load and return the cached file if exists and it was created
    after the original file.

    If ignore_date is True, the file is loaded even if the cache
    is older thant the actual file.
    """
    cache = cachepath_for(original)
    if not cache.exists():
        return
    if ignore_date or cache.stat().st_mtime > original.stat().st_mtime:
        with cache.open(mode="rb") as fi:
            return pickle.load(fi)
    return None


def save(obj, original: pathlib.Path) -> pathlib.Path:
    """Save the object (in pickle format) to the cache folder
    using a unique name generated using `cachepath_for`
    """
    cache = cachepath_for(original)
    with cache.open(mode="wb") as fo:
        pickle.dump(obj, fo)
    return cache
