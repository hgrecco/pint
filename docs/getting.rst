.. _getting:

Installation
============

Pint has no dependencies except Python_ itself. In runs on Python 2.7 and 3.0+.

You can install it using pip_::

    $ pip install pint

or using easy_install_::

    $ easy_install pint

That's all! You can check that Pint is correctly installed by starting up python, and importing pint:

    >>> import pint

.. note:: If you have an old system installation of Python and you don't want to
   mess with it, you can try `Anaconda CE`_. It is a free Python distribution by
   Continuum Analytics that includes many scientific packages.


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
.. _PyPI: https://pypi.python.org/pypi/Pint/
.. _`Anaconda CE`: https://store.continuum.io/cshop/anaconda
.. _GitHub: https://github.com/hgrecco/pint
