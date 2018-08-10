
Pandas support
==============

It is convenient to use pandas when dealing with numerical data, so pint
provides QuantityArray to allow quantities to be used with Pandas. A
QuantityArray is a pandas ExtensionArray, which allows pandas to
recognise the Quantity and store it in DataFrames or Series.
https://pandas.pydata.org/pandas-docs/stable/generated/pandas.api.extensions.ExtensionArray.html

.. code:: ipython3

    import pandas as pd 
    import pint
    import numpy as np
    
    from pint.pandas_array import QuantityArray

.. code:: ipython3

    ureg=pint.UnitRegistry()
    Q_=ureg.Quantity

.. code:: ipython3

    df=pd.DataFrame({"torque":QuantityArray(Q_([1,2,2,3],"lbf ft")),
                  "angular_velocity":QuantityArray(Q_([1000,2000,2000,3000],"rpm"))})
    df




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>torque</th>
          <th>angular_velocity</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1</td>
          <td>1000</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2</td>
          <td>2000</td>
        </tr>
        <tr>
          <th>2</th>
          <td>2</td>
          <td>2000</td>
        </tr>
        <tr>
          <th>3</th>
          <td>3</td>
          <td>3000</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    df['power']=df['torque']*df['angular_velocity']
    df




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>torque</th>
          <th>angular_velocity</th>
          <th>power</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1</td>
          <td>1000</td>
          <td>1000</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2</td>
          <td>2000</td>
          <td>4000</td>
        </tr>
        <tr>
          <th>2</th>
          <td>2</td>
          <td>2000</td>
          <td>4000</td>
        </tr>
        <tr>
          <th>3</th>
          <td>3</td>
          <td>3000</td>
          <td>9000</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    df.power.values.data




.. raw:: html

    \[\begin{pmatrix}1000 & 4000 & 4000 & 9000\end{pmatrix} foot force_pound revolutions_per_minute\]



.. code:: ipython3

    df.power.values.data.to("kW")




.. raw:: html

    \[\begin{pmatrix}0.14198092353610375 & 0.567923694144415 & 0.567923694144415 & 1.2778283118249338\end{pmatrix} kilowatt\]



Comments
--------

What follows is a short discussion about Pint’s QuantityArray Object.

It is first useful to distinguish between three structures stored in a
Pint Quantity: 1. A scalar value

.. code:: ipython3

    Q_(123,"m")

::

   2. A 1d array or list

.. code:: ipython3

    Q_([1,2,3],"m")

::

   3. A 2d+ array or list

.. code:: ipython3

    Q_([[1,2],[3,4]],"m")

A single scalar value is not intended to be stored in the QuantityArray
as it’s not an array, and should raise an error (TODO). The scalar
Quantity is the scalar form of the QuantityArray, and is returned when
performing operations that use **get_item**, eg indexing. A
QuantityArray can be created from a list of scalar Quantitys using
QuantityArray._from_sequence.

The second case is intended to be stored in the QuantityArray, and is
stored in the QuantityArray.data attribute.

ExtensionArrays are limited to 1d arrays, so the third case cannot be
stored in the array, and should raise an error (TODO).

Most operations on the QuantityArray act on the Quantity stored in the
QuantityArray.data, so will behave similiarly to operations on a
Quantity, with some caveats: 1. An operation that would return a 1d
Quantity will return a QuantityArray containing the Quantity. This
allows pandas to assign the result to a Series. 2. Arithemetic and
comparative operations are limited to scalars and sequences of the same
length as the stored Quantity. This ensures results are the same length
as the stored Quantity, so can be added to the same DataFrame.

.. code:: ipython3

    Q_([[1,2,3,4]],"m")*Q_([[1],[2]],"N")




.. raw:: html

    \[\begin{pmatrix}1 & 2 & 3 & 4\\ 
    2 & 4 & 6 & 8\end{pmatrix} meter newton\]


