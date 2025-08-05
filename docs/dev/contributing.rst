.. _contributing:

Contributing to Pint
====================

Pint uses (and thanks):

- github_ to host the code
- `github actions`_ to test all commits and PRs.
- coveralls_ to monitor coverage test coverage
- readthedocs_ to host the documentation.
- black_, isort_ and flake8_ as code linters and pre-commit_ to enforce them.
- pytest_ to write tests
- sphinx_ to write docs.

You can contribute in different ways:

Report issues
-------------

You can report any issues with the package, the documentation to the Pint `issue tracker`_.
Also feel free to submit feature requests, comments or questions.


Contribute code
---------------

To contribute fixes, code or documentation to Pint, fork Pint in github_ and submit
the changes using a pull request against the **master** branch.

- If you are submitting new code, add tests (see below) and documentation.
- Write "Closes #<bug number>" in the PR description or a comment, as described in the
  `github docs`_.
- Log the change in the CHANGES file.
- Execute ``pre-commit run --all-files`` and resolve any issues.

In any case, feel free to use the `issue tracker`_ to discuss ideas for new features or improvements.

Notice that we will not merge a PR if tests are failing. In certain cases tests pass in your
machine but not in travis. There might be multiple reasons for this but these are some of
the most common

- Your new code does not work for other Python or Numpy versions.
- The documentation is not being built properly or the examples in the docs are
  not working.
- linters are reporting that the code does no adhere to the standards.


Setting up your environment
---------------------------

Here is how to set up your environment if you're contributing to this project for the fist time.

Using pixi
~~~~~~~~~~

First install `pixi`_, then run the following commands in a terminal

    $ git clone git@github.com:hgrecco/pint.git
    $ cd pint

pixi handles setting up environments and downloading and installing modules so you don't neeed to
run any commands to install pint or its dependencies, you just need to be in the pint directory.


To run the tests, linting and documentation building, you can use the following commands:

  $ pixi run --environment test-py313-all test # runs tests only
  $ pixi run --environment test-py313-all pytest -k test_log # run specific tests
  $ pixi run --environment test-py313-all bench # runs tests and benchmarks. Much slower and rarely needed.
  $ pixi run --environment lint lint --all-files
  $ pixi run docbuild
  $ pixi run doctest

Other common commands include:

  $ pixi run --environment test-py313-all python # runs a python shell with pint installed
  $ pixi run --environment test-py313-all python my_script.py # runs a script with pint installed

A full list of the environments and commands available can be found in the `pyproject.toml`_ file.


Setting up your environment (legacy)
------------------------------------

It is recommended to use pixi, as described above. Alternatively if you want to set up
your environment manually, you can do so using `venv`/`pip` or `conda`/`pip`.

Using venv/pip with Linux or OSX
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In a terminal, navigate where you want the code to be downloaded, and type::

    $ git clone git@github.com:hgrecco/pint.git
    $ cd pint
    $ python -m virtualenv venv
    $ source venv/bin/activate
    $ pip install -e '.[test]'
    $ pip install -r requirements_docs.txt
    $ pip install pre-commit # This step and the next are optional but recommended.
    $ pre-commit install

Using conda/pip with Windows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here it is assumed you are working with Conda, and that Pint has already been git-cloned.
In an Anaconda Prompt, activate your environment and navigate to the Pint directory.
Then type::

    $ conda install pip
    $ conda install git
    $ pip install -e ".[test]"
    $ pip install -r requirements_docs.txt
    $ pip install pre-commit
    $ pre-commit install

All
~~~

Additionally, the following external applications are required at some stage:

- Graphviz_ (used in the `visualize()` function of the Dask library)
- Pandoc_ (used in documentation building)
- Pixi_ (used when committing changes)

Writing tests
-------------

We use pytest_ for testing. If you contribute code you need to add tests:

- If you are fixing a bug, add a test to `test_issues.py`, or amend/enrich the general
  test suite to cover the use case.
- If you are adding a new feature, add a test in the appropiate place. There is usually
  a `test_X.py` for each `X.py` file. There are some other test files that deal with
  individual/specific features. If in doubt, ask.
- Prefer functions to classes.
- When using classes, derive from `QuantityTestCase`.
- Use `parametrize` as much as possible.
- Use `fixtures` (see conftest.py) instead of instantiating the registry yourself.
  Check out the existing fixtures before creating your own.
- When your test does not modify the registry, use `sess_registry` fixture.
- **Do not** create a unit registry outside a test or fixture setup.
- If you need a specific registry, and you need to reuse it create a
  fixture in your test module called `local_registry` or similar.
- Checkout `helpers.py` for some convenience functions before reinventing the wheel.


Running tests and building documentation
----------------------------------------

To run the test suite, invoke pytest from the ``pint`` directory from a terminal with
administrator privileges (otherwise you may run into permission errors due to temporary
file management)::

    $ cd pint
    $ pytest

To run the doctests, invoke Sphinx's doctest module from the ``docs`` directory::

    $ cd docs
    $ make doctest

To build the documentation, invoke Sphinx from the ``docs`` directory::

    $ cd docs
    $ make html

Extension Packages
------------------

Pint naturally integrates with other libraries in the scientific Python ecosystem, and
a small _`ecosystem` have arisen to aid in compatibility between certain packages
allowing to build an

Pint's rule of thumb for integration
features that work best as an extension package versus direct inclusion in Pint is:

* Extension (separate packages)

  * Duck array types that wrap Pint (come above Pint
    in :ref:`the type casting hierarchy <_numpy#technical-commentary>`)

  * Uses features independent/on top of the libraries

  * Examples: xarray, Pandas

* Integration (built in to Pint)

  * Duck array types wrapped by Pint (below Pint in the type casting hierarchy)

  * Intermingling of APIs occurs

  * Examples: Dask


Creating a release
------------------

Maintainers may create a new release by tagging a commit::

    $ # do changes and commit
    $ git tag -a 0.24.rc0 -m "Tagging 0.24.rc0"
    $ git push --tags

For the final release, add date to the 0.24 section in CHANGES, then::

    $ git commit -a -m "Preparing for release 0.24"
    $ git tag -a 0.24 -m "Tagging 0.24"

Then add to CHANGES the following::

    0.25 (unreleased)
    -----------------

And push the tags and CHANGES ::

    $ git commit -a -m "Back to development: 0.25"
    $ git push --tags



.. _github: http://github.com/hgrecco/pint
.. _`issue tracker`: https://github.com/hgrecco/pint/issues
.. _`github docs`: https://help.github.com/articles/closing-issues-via-commit-messages/
.. _`github actions`: https://docs.github.com/en/actions
.. _coveralls: https://coveralls.io/
.. _readthedocs: https://readthedocs.org/
.. _pre-commit: https://pre-commit.com/
.. _black: https://black.readthedocs.io/en/stable/
.. _isort: https://pycqa.github.io/isort/
.. _flake8: https://flake8.pycqa.org/en/latest/
.. _pytest: https://docs.pytest.org/en/stable/
.. _sphinx: https://www.sphinx-doc.org/en/master/
.. _`extension/compatibility packages`:
.. _Graphviz: https://graphviz.gitlab.io/download/
.. _Pandoc: https://pandoc.org/installing.html
.. _Pixi: https://pixi.sh/latest/installation/
.. _pyproject.toml: https://github.com/hgrecco/pint/blob/master/pyproject.toml
