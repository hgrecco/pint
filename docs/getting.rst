.. _getting:

Installation
============

Pint has no dependencies except Python_ itself. In runs on Python 3.8+.

You can install it (or upgrade to the latest version) using pip_::

    $ pip install -U pint

That's all! You can check that Pint is correctly installed by starting up python, and importing Pint:

.. code-block:: python

    >>> import pint
    >>> pint.__version__  # doctest: +SKIP

Or running the test suite:

.. code-block:: python

    >>> pint.test()

.. note:: If you have an old system installation of Python and you don't want to
   mess with it, you can try `Anaconda CE`_. It is a free Python distribution by
   Continuum Analytics that includes many scientific packages. To install pint
   from the conda-forge channel instead of through pip use::

       $ conda install -c conda-forge pint


Getting the code
----------------

You can also get the code from PyPI_ or GitHub_. You can either clone the public repository::

    $ git clone git://github.com/hgrecco/pint.git

Download the tarball::

    $ curl -OL https://github.com/hgrecco/pint/tarball/master

Or, download the zipball::

    $ curl -OL https://github.com/hgrecco/pint/zipball/master

Once you have a copy of the source, you can embed it in your Python package, or install it into your site-packages easily::

    $ python setup.py install


.. _easy_install: http://pypi.python.org/pypi/setuptools
.. _Python: http://www.python.org/
.. _pip: http://www.pip-installer.org/
.. _`Anaconda CE`: https://store.continuum.io/cshop/anaconda
.. _PyPI: https://pypi.python.org/pypi/Pint/
.. _GitHub: https://github.com/hgrecco/pint
