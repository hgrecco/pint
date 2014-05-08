.. _getting:

Installation
============

Pint has no dependencies except Python_ itself. In runs on Python 2.6 and 3.0+.

You can install it using pip_::

    $ pip install pint

or using easy_install_::

    $ easy_install pint

That's all! You can check that Pint is correctly installed by starting up python, and importing pint:

    >>> import pint
    >>> pint.__version__
    0.5

.. note:: If you have an old system installation of Python and you don't want to
   mess with it, you can try `Anaconda CE`_. It is a free Python distribution by
   Continuum Analytics that includes many scientific packages.

You can check the installation with the following command:

    >>> pint.test()


On Arch Linux, you can alternatively install Pint from the Arch User Repository
(AUR). The latest release is available as `python-pint`_, and packages tracking
the master branch of the GitHub repository are available as `python-pint-git`_
and `python2-pint-git`_.


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
.. _`python-pint`: https://aur.archlinux.org/packages/python-pint/
.. _`python-pint-git`: https://aur.archlinux.org/packages/python-pint-git/
.. _`python2-pint-git`: https://aur.archlinux.org/packages/python2-pint-git/
.. _PyPI: https://pypi.python.org/pypi/Pint/
.. _GitHub: https://github.com/hgrecco/pint
