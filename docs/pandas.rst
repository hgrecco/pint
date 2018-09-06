.. _pandas:

Pandas support
==============

It is convenient to use the `Pandas package`_ when dealing with numerical data, so Pint provides `PintArray`. A `PintArray` is a `Pandas Extension Array`_, which allows Pandas to recognise the Quantity and store it in Pandas DataFrames and Series.

For this to work, we rely on `Pandas Extension Types`_ which are still experimental. As a result, we currently have to build the latest version of Pandas' master branch from source as documented in the `Pandas README`_.

Basic example
-------------

This example gives you the basics, but is slightly fiddly as you are not reading from a file. A more normal use case is given in `Reading a csv`_.

To use Pint with Pandas, as stated above, firstly ensure that you have the latest version of Pandas installed. Then import the relevant packages and create an instance of a Pint Quantity:

.. doctest::

   >>> import pandas as pd
   >>> import numpy as np
   >>> import pint
   >>> from pint.pandas_interface import PintArray

   >>> ureg = pint.UnitRegistry()
   >>> Q_ = ureg.Quantity

.. testsetup:: *

   import pandas as pd
   import numpy as np
   import pint
   from pint.pandas_interface import PintArray

   ureg = pint.UnitRegistry()
   Q_ = ureg.Quantity

Next, we can create a DataFrame with PintArray's as columns

.. doctest::

   >>> torque = PintArray(Q_([1, 2, 2, 3], "lbf ft"))
   >>> angular_velocity = PintArray(Q_([1000, 2000, 2000, 3000], "rpm"))
   >>> df = pd.DataFrame({"torque": torque, "angular_velocity": angular_velocity})
   >>> print(df)
    torque angular_velocity
   0      1             1000
   1      2             2000
   2      2             2000
   3      3             3000


Operations with columns are units aware so behave as we would intuitively expect

.. doctest::

   >>> df['power'] = df['torque'] * df['angular_velocity']
   >>> print(df)
     torque angular_velocity power
   0      1             1000  1000
   1      2             2000  4000
   2      2             2000  4000
   3      3             3000  9000


Each column can be accessed as a Pandas Series

.. doctest::

   >>> print(df.power)
   0    1000
   1    4000
   2    4000
   3    9000
   Name: power, dtype: pint


Which contains a PintArray

.. doctest::

   >>> print(df.power.values)
    PintArray([1000 foot * force_pound * revolutions_per_minute,
               4000 foot * force_pound * revolutions_per_minute,
               4000 foot * force_pound * revolutions_per_minute,
               9000 foot * force_pound * revolutions_per_minute],
              dtype='pint')


Which contains a Quantity

.. doctest::

   >>> print(df.power.values.data)
   [1000 4000 4000 9000] foot * force_pound * revolutions_per_minute


Pandas Series accessors are provided for most Quantity properties and methods, which will convert the result to a Series where possible.

.. doctest::

   >>> print(df.power.pint.dimensionality)
   [length] ** 2 * [mass] / [time] ** 3

   >>> print(df.power.pint.to("kW"))
   0    0.14198092353610375
   1      0.567923694144415
   2      0.567923694144415
   3     1.2778283118249338
   Name: power, dtype: pint


Standard pint conversions can still be performed on the underlying quantity, and will still return a quantity.

.. doctest::

   >>> print(df.power.values.data.to("kW"))
   [0.14198092 0.56792369 0.56792369 1.27782831] kilowatt

Reading a csv
-------------

Thanks to the DataFrame accessors, reading from files with unit information becomes trivial. The DataFrame accessors make it easy to get to PintArrays.

Setup
~~~~~

Here we create the DateFrame and save it to file, next we will show you how to load and read it.

We start with an DateFrame with column headers only.

.. doctest::

   >>> speed = [1000, 1100, 1200, 1200]
   >>> mech_power = [np.nan, np.nan, np.nan, np.nan]
   >>> torque = [10, 10, 10, 10]
   >>> rail_pressure = [1000, 1000000000000, 1000, 1000]
   >>> fuel_flow_rate = [10, 10, 10, 10]
   >>> fluid_power = [np.nan, np.nan, np.nan, np.nan]
   >>> df_init = pd.DataFrame({"speed": speed, "mech power": mech_power, "torque": torque, "rail pressure": rail_pressure, "fuel flow rate": fuel_flow_rate, "fluid power": fluid_power,})
   >>> print(df_init)
      speed  mech power  torque  rail pressure  fuel flow rate  fluid power
   0   1000         NaN      10           1000              10          NaN
   1   1100         NaN      10  1000000000000              10          NaN
   2   1200         NaN      10           1000              10          NaN
   3   1200         NaN      10           1000              10          NaN

Then we add a column header which contains units information

.. doctest::

   >>> units = ["rpm", "kW", "N m", "bar", "l/min", "kW"]
   >>> df_to_save = df_init.copy()
   >>> df_to_save.columns = pd.MultiIndex.from_arrays([df_init.columns, units])
   >>> print(df_to_save)
     speed mech power torque  rail pressure fuel flow rate fluid power
       rpm         kW    N m            bar          l/min          kW
   0  1000        NaN     10           1000             10         NaN
   1  1100        NaN     10  1000000000000             10         NaN
   2  1200        NaN     10           1000             10         NaN
   3  1200        NaN     10           1000             10         NaN

Now we save this to disk as a csv to give us our starting point.

.. doctest::

   >>> test_csv_name = "pandas_test.csv"
   >>> df_to_save.to_csv(test_csv_name, index=False)

Now we are in a position to read the csv we just saved. Let's start by reading the file with units as a level in a multiindex column.

.. doctest::

   >>> df = pd.read_csv(test_csv_name, header=[0,1])
   >>> print(df)
     speed mech power torque  rail pressure fuel flow rate fluid power
       rpm         kW    N m            bar          l/min          kW
   0  1000        NaN     10           1000             10         NaN
   1  1100        NaN     10  1000000000000             10         NaN
   2  1200        NaN     10           1000             10         NaN
   3  1200        NaN     10           1000             10         NaN

Then use the DataFrame's `pint.quantify` method to convert the columns from `np.ndarray`s to PintArrays, with units from the bottom column level.

.. doctest::

   >>> df_ = df.pint.quantify(ureg, level=-1)
   >>> print(df_)
       speed mech power torque    rail pressure fuel flow rate fluid power
   0  1000.0        nan   10.0           1000.0           10.0         nan
   1  1100.0        nan   10.0  1000000000000.0           10.0         nan
   2  1200.0        nan   10.0           1000.0           10.0         nan
   3  1200.0        nan   10.0           1000.0           10.0         nan


As previously, operations between DataFrame columns are unit aware

.. doctest::

   >>> df_['mech power'] = df_.speed*df_.torque
   >>> df_['fluid power'] = df_['fuel flow rate'] * df_['rail pressure']
   >>> print(df_)
       speed mech power torque    rail pressure fuel flow rate       fluid power
   0  1000.0    10000.0   10.0           1000.0           10.0           10000.0
   1  1100.0    11000.0   10.0  1000000000000.0           10.0  10000000000000.0
   2  1200.0    12000.0   10.0           1000.0           10.0           10000.0
   3  1200.0    12000.0   10.0           1000.0           10.0           10000.0


The DataFrame's `pint.dequantify` method then allows us to retrieve the units information as a header row once again

.. doctest::

   >>> print(df_.pint.dequantify())
                      speed                              mech power  \
     revolutions_per_minute meter * newton * revolutions_per_minute
   0                 1000.0                                 10000.0
   1                 1100.0                                 11000.0
   2                 1200.0                                 12000.0
   3                 1200.0                                 12000.0

             torque rail pressure fuel flow rate          fluid power
     meter * newton           bar liter / minute bar * liter / minute
   0           10.0  1.000000e+03           10.0         1.000000e+04
   1           10.0  1.000000e+12           10.0         1.000000e+13
   2           10.0  1.000000e+03           10.0         1.000000e+04
   3           10.0  1.000000e+03           10.0         1.000000e+04



This allows for some rather powerful abilities. For example, to change single column units

.. doctest::

   >>> df_['fluid power'] = df_['fluid power'].pint.to("kW")
   >>> df_['mech power'] = df_['mech power'].pint.to("kW")
   >>> print(df_.pint.dequantify())

                      speed mech power         torque rail pressure  \
     revolutions_per_minute   kilowatt meter * newton           bar
   0                 1000.0   1.047198           10.0  1.000000e+03
   1                 1100.0   1.151917           10.0  1.000000e+12
   2                 1200.0   1.256637           10.0  1.000000e+03
   3                 1200.0   1.256637           10.0  1.000000e+03

     fuel flow rate   fluid power
     liter / minute      kilowatt
   0           10.0  1.666667e+01
   1           10.0  1.666667e+10
   2           10.0  1.666667e+01
   3           10.0  1.666667e+01


or the entire table's units

.. doctest::

   >>> print(df_.pint.to_base_units().pint.dequantify())

               speed                          mech power  \
     radian / second kilogram * meter ** 2 / second ** 3
   0      104.719755                         1047.197551
   1      115.191731                         1151.917306
   2      125.663706                         1256.637061
   3      125.663706                         1256.637061

                                  torque                  rail pressure  \
     kilogram * meter ** 2 / second ** 2 kilogram / meter / second ** 2
   0                                10.0                   1.000000e+08
   1                                10.0                   1.000000e+17
   2                                10.0                   1.000000e+08
   3                                10.0                   1.000000e+08

          fuel flow rate                         fluid power
     meter ** 3 / second kilogram * meter ** 2 / second ** 3
   0            0.000167                        1.666667e+04
   1            0.000167                        1.666667e+13
   2            0.000167                        1.666667e+04
   3            0.000167                        1.666667e+04


Comments
--------

What follows is a short discussion about Pint's `PintArray` Object.

It is first useful to distinguish between three different things:

1. A scalar value

.. doctest::

   >>> print(Q_(123,"m"))
   123 meter

2. A scalar value

.. doctest::

   >>> print(Q_([1, 2, 3], "m"))
   [1 2 3] meter

3. A scalar value

.. doctest::

   >>> print(Q_([[1, 2], [3, 4]], "m"))
   [[1 2] [3 4]] meter


The first, a single scalar value is not intended to be stored in the PintArray as it's not an array, and should raise an error (TODO). The scalar Quantity is the scalar form of the PintArray, and is returned when performing operations that use `get_item`, eg indexing. A PintArray can be created from a list of scalar Quantitys using `PintArray._from_sequence`.

The second, a 1d array or list, is intended to be stored in the PintArray, and is stored in the PintArray.data attribute.

The third, 2d+ arrays or lists, are beyond the capabilities of ExtensionArrays which are limited to 1d arrays, so cannot be stored in the array, and should raise an error (TODO).

Most operations on the PintArray act on the Quantity stored in `PintArray.data`, so will behave similiarly to operations on a Quantity, with some caveats:

1. An operation that would return a 1d Quantity will return a PintArray containing the Quantity. This allows pandas to assign the result to a Series.
2. Arithemetic and comparative operations are limited to scalars and sequences of the same length as the stored Quantity. This ensures results are the same length as the stored Quantity, so can be added to the same DataFrame.




.. _`Pandas package`: https://pandas.pydata.org/pandas-docs/stable/index.html
.. _`Pandas Dataframes`: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html
.. _`Pandas Extension Array`: https://pandas.pydata.org/pandas-docs/stable/extending.html#extensionarray
.. _`Pandas Extension Types`: https://pandas.pydata.org/pandas-docs/stable/extending.html#extension-types
.. _`Pandas README`: https://github.com/pandas-dev/pandas/blob/master/README.md

