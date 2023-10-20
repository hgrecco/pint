In Pint, we work hard to avoid breaking projects that depend on us.
If you are the maintainer of one of such projects, you can
help us get ahead of problems in simple way.

Pint will publish a release candidate (rc) at least a week before each new
version. By default, `pip` does not install these versions unless a
[pre](https://pip.pypa.io/en/stable/cli/pip_install/#cmdoption-pre) option
is used so this will not affect your users.

In addition to your standard CI routines, create a CI that install Pint's
release candidates. You can also (or alternatively) create CI that install
Pint's master branch in GitHub.

Take a look at the [Pint Downstream Demo](https://github.com/hgrecco/pint-downstream-demo)
if you need a template.

Then, add your project badges to this file so it can be used as a Dashboard (always putting the stable first)

| Project                                                                 | stable                                                                                                                                                                | pre-release                                                                                                                                                                                      | nightly                                                                                                                                                                                                   |
| ----------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Pint Downstream Demo](https://github.com/hgrecco/pint-downstream-demo) | [![CI](https://github.com/hgrecco/pint-downstream-demo/actions/workflows/ci.yml/badge.svg)](https://github.com/hgrecco/pint-downstream-demo/actions/workflows/ci.yml) | [![CI-pint-pre](https://github.com/hgrecco/pint-downstream-demo/actions/workflows/ci-pint-pre.yml/badge.svg)](https://github.com/hgrecco/pint-downstream-demo/actions/workflows/ci-pint-pre.yml) | [![CI-pint-master](https://github.com/hgrecco/pint-downstream-demo/actions/workflows/ci-pint-master.yml/badge.svg)](https://github.com/hgrecco/pint-downstream-demo/actions/workflows/ci-pint-master.yml) |
| [Pint Pandas](https://github.com/hgrecco/pint-pandas)                   | [![CI](https://github.com/hgrecco/pint-pandas/actions/workflows/ci.yml/badge.svg)](https://github.com/hgrecco/pint-pandas/actions/workflows/ci.yml)                   | [![CI-pint-pre](https://github.com/hgrecco/pint-pandas/actions/workflows/ci-pint-pre.yml/badge.svg)](https://github.com/hgrecco/pint-pandas/actions/workflows/ci-pint-pre.yml)                   | [![CI-pint-master](https://github.com/hgrecco/pint-pandas/actions/workflows/ci-pint-master.yml/badge.svg)](https://github.com/hgrecco/pint-pandas/actions/workflows/ci-pint-master.yml)                   |
| [MetPy](https://github.com/Unidata/MetPy)                               | [![CI](https://github.com/Unidata/MetPy/actions/workflows/tests-pypi.yml/badge.svg)](https://github.com/Unidata/MetPy/actions/workflows/tests-pypi.yml)               |                                                                                                                                                                                                  | [![CI-pint-master](https://github.com/Unidata/MetPy/actions/workflows/nightly-builds.yml/badge.svg)](https://github.com/Unidata/MetPy/actions/workflows/nightly-builds.yml)                               |
| [pint-xarray](https://github.com/xarray-contrib/pint-xarray)            | [![CI](https://github.com/xarray-contrib/pint-xarray/actions/workflows/ci.yml/badge.svg)](https://github.com/xarray-contrib/pint-xarray/actions/workflows/ci.yml)     |                                                                                                                                                                                                  | [![CI-pint-master](https://github.com/xarray-contrib/pint-xarray/actions/workflows/nightly.yml/badge.svg)](https://github.com/xarray-contrib/pint-xarray/actions/workflows/nightly.yml)                   |
