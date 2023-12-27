Getting Started
===============

The getting started guide aims to get you using pint productively as quickly as possible.



Installation
------------

Pint has no dependencies except Python itself. It runs on Python 3.9+.

.. grid:: 2

    .. grid-item-card::  Prefer pip?

        **pint** can be installed via pip from `PyPI <https://pypi.org/project/pint>`__.

        ++++++++++++++++++++++

        .. code-block:: bash

            pip install pint

    .. grid-item-card::  Working with conda?

        **pint** is part of the `Conda-Forge <https://conda-forge.org//>`__
        channel and can be installed with Anaconda or Miniconda:

        ++++++++++++++++++++++

        .. code-block:: bash

            conda install -c conda-forge pint


That's all! You can check that Pint is correctly installed by starting up python, and importing Pint:

.. code-block:: python

    >>> import pint
    >>> pint.__version__  # doctest: +SKIP

.. toctree::
    :maxdepth: 2
    :hidden:

    overview
    tutorial
    pint-in-your-projects
    faq
