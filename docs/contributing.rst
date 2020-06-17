.. _contributing:

Contributing to Pint
====================

You can contribute in different ways:


Report issues
-------------

You can report any issues with the package, the documentation to the Pint `issue tracker`_.
Also feel free to submit feature requests, comments or questions.


Contribute code
---------------

To contribute fixes, code or documentation to Pint, fork Pint in github_ and submit
the changes using a pull request against the **master** branch.

- If you are fixing a bug, add a test to test_issues.py, or amend/enrich the general
  test suite to cover the use case.
- If you are submitting new code, add tests and documentation.
- Write "Closes #<bug number>" in the PR description or a comment, as described in the
  `github docs`_.
- Log the change in the CHANGES file.
- Execute ``black -t py36 . && isort -rc . && flake8`` and resolve any issues.

Pint uses `bors-ng` as a merge bot and therefore every PR is tested before merging.

In any case, feel free to use the `issue tracker`_ to discuss ideas for new features or improvements.


Setting up your environment
---------------------------

If you're contributing to this project for the fist time, you can set up your
environment on Linux or OSX with the following commands::

    $ git clone git@github.com:hgrecco/pint.git
    $ cd pint
    $ python -m virtualenv venv
    $ source venv/bin/activate
    $ pip install -e .
    $ pip install -r requirements_docs.txt

Running tests and building documentation
----------------------------------------

To run the test suite, invoke pytest from the ``pint`` directory::

    $ cd pint
    $ pytest

To run the doctests, invoke Sphinx's doctest module from the ``docs`` directory::

    $ cd docs
    $ make doctest

To build the documentation, invoke Sphinx from the ``docs`` directory::

    $ cd docs
    $ make html


.. _github: http://github.com/hgrecco/pint
.. _`issue tracker`: https://github.com/hgrecco/pint/issues
.. _`bors-ng`: https://github.com/bors-ng/bors-ng
.. _`github docs`: https://help.github.com/articles/closing-issues-via-commit-messages/