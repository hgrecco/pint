In Pint, we work hard to avoid breaking projects that depend on us.
If you are the maintainer of one of such projects, you can
help us get ahead of problems in simple way.

In addition to your standard CI routines, create a CI that install Pint's
release candidates. You can also (or alternatively) create CI that install
Pint's master branch in GitHub.

Take a look at the [Pint Downstream Demo](https://github.com/hgrecco/pint-downstream-demo)
if you need a template.

Then, add your project badges to this file so it can be used as a Dashboard (always putting the stable first)

[Pint Downstream Demo](https://github.com/hgrecco/pint-downstream-demo)
[![CI](https://github.com/hgrecco/pint-downstream-demo/actions/workflows/ci.yml/badge.svg)](https://github.com/hgrecco/pint-downstream-demo/actions/workflows/ci.yml)
[![CI-pint-pre](https://github.com/hgrecco/pint-downstream-demo/actions/workflows/ci-pint-pre.yml/badge.svg)](https://github.com/hgrecco/pint-downstream-demo/actions/workflows/ci-pint-pre.yml)
[![CI-master](https://github.com/hgrecco/pint-downstream-demo/actions/workflows/ci-master.yml/badge.svg)](https://github.com/hgrecco/pint-downstream-demo/actions/workflows/ci-master.yml)
