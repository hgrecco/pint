.. _performance:


Optimizing Performance
======================

Pint can impose a significant performance overhead on computationally-intensive problems. The following are some suggestions for getting the best performance.

.. note:: Examples below are based on the IPython shell (which provides the handy %timeit extension), so they will not work in a standard Python interpreter.

Use magnitudes when possible
----------------------------

It's significantly faster to perform mathematical operations on magnitudes (even though your'e still using pint to retrieve them from a quantity object).

.. doctest::

   In [1]: from pint import UnitRegistry

   In [2]: ureg = UnitRegistry()

   In [3]: q1 =ureg('1m')

   In [5]: q2=ureg('2m')

   In [6]: %timeit (q1-q2)
   100000 loops, best of 3: 7.9 µs per loop

   In [7]: %timeit (q1.magnitude-q2.magnitude)
   1000000 loops, best of 3: 356 ns per loop

This is especially important when using pint Quantities in conjunction with an iterative solver, such as the `brentq method`_ from scipy:

.. doctest::

    In [1]: from scipy.optimize import brentq

    In [2]: def foobar_with_quantity(x):
                # find the value of x that equals q2
            
                # assign x the same units as q2
                qx = ureg(str(x)+str(q2.units))
        
                # compare the two quantities, then take their magnitude because
                # brentq requires a dimensionless return type
                return (qx - q2).magnitude
        
    In [3]: def foobar_with_magnitude(x):
                # find the value of x that equals q2
    
                # don't bother converting x to a quantity, just compare it with q2's magnitude
                return x - q2.magnitude
    
    In [4]: %timeit brentq(foobar_with_quantity,0,q2.magnitude)
    1000 loops, best of 3: 310 µs per loop
    
    In [5]: %timeit brentq(foobar_with_magnitude,0,q2.magnitude)
    1000000 loops, best of 3: 1.63 µs per loop

Bear in mind that altering computations like this **loses the benefits of automatic unit conversion**, so use with care.

A safer method: wrapping
------------------------
A better way to use magnitudes is to use pint's wraps decorator (See :ref:`wrapping`). By decorating a function with wraps, you pass only the magnitude fo an argument to the function body according to units you specify. As such this method is safer in that you are sure the magnitude is supplied in the correct units.

.. doctest::

    In [1]: import pint

    In [2]: ureg = pint.UnitRegistry()

    In [3]: import numpy as np

    In [4]: def f(x, y):
                  return (x - y) / (x + y) * np.log(x/y)

    In [5]: @ureg.wraps(None, ('meter', 'meter'))
             def g(x, y):
                 return (x - y) / (x + y) * np.log(x/y)

    In [6]: a = 1 * ureg.meter

    In [7]: b = 1 * ureg.centimeter

    In [8]: %timeit f(a, b)
    1000 loops, best of 3: 312 µs per loop

    In [9]: %timeit g(a, b)
    10000 loops, best of 3: 65.4 µs per loop

.. _`brentq method`: http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brentq.html