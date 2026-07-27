Try Pint in your browser
=========================

You can try Pint directly in your browser, with no installation required. The
console and notebook below run entirely client-side using `JupyterLite
<https://jupyterlite.readthedocs.io/>`__ and `Pyodide
<https://pyodide.org/>`__ — your code never leaves your machine.

The first time you run a cell, it may take a few seconds to start up the
in-browser Python environment.

Quick console
-------------

Type or paste code below and press ``Shift+Enter`` to run it (the "Run"
button in the toolbar also works). This snippet installs Pint and computes a
short unit conversion to get you started:

.. replite::
   :kernel: python
   :height: 600px

   %pip install -q pint
   import pint

   ureg = pint.UnitRegistry()
   distance = 42 * ureg.kilometer
   print(distance.to("mile"))

Full notebook
-------------

For a longer, guided walkthrough covering Pint's NumPy support, open the
notebook below:

.. notebooklite:: try_pint_numpy.ipynb
   :width: 100%
   :height: 700px
