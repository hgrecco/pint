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

- If you are fixing a bug, add a test to test_issues.py
  Also add "Close #<bug number> as described in the `github docs`_.
- If you are submitting new code, add tests and documentation.
- If adding or modifying a feature, make sure it's documented in docs/.
- Log the change in the CHANGES file.
- Execute ``black -t py36 . && isort -rc . && flake8`` and resolve any issues

Pint uses `bors-ng` as a merge bot and therefore every PR is tested before merging.

In any case, feel free to use the `issue tracker`_ to discuss ideas for new features or improvements.


.. _github: http://github.com/hgrecco/pint
.. _`issue tracker`: https://github.com/hgrecco/pint/issues
.. _`bors-ng`: https://github.com/bors-ng/bors-ng
.. _`github docs`: https://help.github.com/articles/closing-issues-via-commit-messages/