"""
Stochastic Math Package
Implements classes and methods for intrusive calculations of random variables with first and second moment estimation.

"""

__author__ = 'Alex Feldstein <alexfeldstein1@gmail.com>'


import autograd.numpy as np
from autograd import grad, hessian


def _mean_var_from_derivs(variables, mean, gradient, hess):
    """
    This method estimates the mean and variance of a function of random variables through a Taylor series expansion.

    :param variables: list of RandomVariable instances used in the function
    :param mean: mean of the function as computed using the means of the RVs
    :param gradient: gradient of the function with respect to the RVs
    :param hess: hessian of the function with respect to the RVs
    :return: new RandomVariable instance
    """
    cov = covariance_matrix(variables)

    mean_off = 0.5 * np.trace(hess.dot(cov))
    mean += mean_off
    variance = gradient.dot(cov).dot(gradient) + 0.5 * np.trace(hess.dot(cov).dot(hess.dot(cov)))

    # modify outputs of 1D case. Bug in autograd?
    if isinstance(mean, np.ndarray) and mean.size == 1:
        mean = mean[0]
        variance = variance[0][0]

    newrv = RandomVariable(mean, variance**0.5)

    n = len(variables)
    cov_newrv_v = gradient.dot(cov)

    # store covariances in the covariance dictionary
    for i in range(n):
        v = variables[i]
        if isinstance(v, RandomVariable):
            newrv._covariance_dict.update({v: cov_newrv_v[i]})
            v._covariance_dict.update({newrv: cov_newrv_v[i]})

            newrv._partial_derivs.update({v: gradient[i]})

    return newrv


def nonlinear_RVmath(func, variables):
    """
    Computes a function of Random variables
    :param func: function object
    :param variables: list of RandomVariable objects
    :return: result, RandomVariable instance
    """
    x = np.array([])
    v = np.array([])
    for var in variables:
        if isinstance(var, RandomVariable):
            x = np.append(x, var._mean)
            v = np.append(v, var._variance)
        else:
            raise TypeError("Currently only RandomVariable instances can be used with this function")

    # use autograd functions to auto-differentiate the function
    f = func(x)
    g = grad(func)(x)
    H = hessian(func)(x)

    return _mean_var_from_derivs(variables, f, g, H)


def covariance_matrix(rvs_in):
    """
    Compute the covariance matrix of a set of RandomVariables
    :param rvs_in: list of variables, can be Quantities, Measurements, or RandomVariables
    :return: covariance matrix as numpy array
    """

    # have to put these imports here or the import will fail
    from ..measurement import _Measurement
    from ..quantity import _Quantity

    # handle case of input RVs being Measurements
    rvs = []
    for rv in rvs_in:
        if isinstance(rv, (_Measurement, _Quantity)):
            rvs.append(rv._magnitude)
        else:
            rvs.append(rv)

    n = len(rvs)
    cov = np.zeros([n, n])

    for i in range(n):
        rvi = rvs[i]

        if isinstance(rvi, RandomVariable):
            cov[i, i] = rvi._variance
        else:
            cov[i, i] = 0.0
            continue

        for j in range(i+1, n):
            rvj = rvs[j]
            if not isinstance(rvj, RandomVariable):
                cov[i, j] = 0.0

            elif rvj in rvi._covariance_dict:
                cov[i, j] = rvi._covariance_dict[rvj]

            elif len(rvj._partial_derivs) > 1:
                pderiv_vars = [v for v in rvj._partial_derivs if v is not rvj]
                tmp1 = [rvj._partial_derivs[v] * covariance_matrix([v, rvi])[0, 1] for v in pderiv_vars]

                cov[i, j] = sum(tmp1)

            elif len(rvi._partial_derivs) > 1:
                pderiv_vars = [v for v in rvi._partial_derivs if v is not rvi]
                tmp1 = [rvi._partial_derivs[v] * covariance_matrix([v, rvj])[0, 1] for v in pderiv_vars]

                cov[i, j] = sum(tmp1)

            else:
                cov[i, j] = 0.0

            cov[j, i] = cov[i, j]

    return cov


def rvs_from_covariance(means, covmatrix):
    """
    Generates a list of RandomVariable instances from a list of means and a covariance matrix
    :param means: list of mean values
    :param covmatrix: numpy covariance matrix
    :return: list of RandomVariable instances
    """
    n = len(covmatrix)

    rvs = [RandomVariable(means[i], covmatrix[i, i]**0.5) for i in range(n)]

    for i in range(n):
        rvnow = rvs[i]
        for j in range(i+1, n):
            rvnow._covariance_dict.update({rvs[j]: covmatrix[i, j]})
            rvs[j]._covariance_dict.update({rvnow: covmatrix[i, j]})

    return rvs


class RandomVariable(object):
    """
    RandomVariable allows for computation of random quantities in an intuitive manner. This class behaves like a number class.
    """
    def __init__(self, mean, std_dev=None):
        """
        RandomVariable allows for computation of random quantities in an intuitive manner. This class behaves like a number class.
        :param mean: mean value of RV
        :param std_dev: standard deviation of RV
        """
        self._mean = mean
        if std_dev is not None:
            self._variance = std_dev**2
        else:
            self._variance = None

        self._partial_derivs = {self: 1.0}
        self._covariance_dict = {self: self._variance}

    def __str__(self):
        return "{0:.4f} w/ sigma {1:.2f}".format(self._mean, self._variance**0.5)

    # necessary for some of the wrapped numpy functions
    def __call__(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        :return: mean of RV
        """
        return self._mean
        
    def __float__(self):
        return float(self._mean)
    
    def __int__(self):
        return int(self._mean)

    @property
    def variance(self):
        return self._variance

    @property
    def nominal_value(self):
        return self._mean

    @property
    def std_dev(self):
        return self._variance**0.5

    def sin(self):
        x = self._mean
        g = np.array([np.cos(x)])
        H = np.array([[-np.sin(x)]])
        return _mean_var_from_derivs([self], np.sin(x), g, H)

    def cos(self):
        x = self._mean
        g = np.array([-np.sin(x)])
        H = np.array([[-np.cos(x)]])
        return _mean_var_from_derivs([self], np.cos(x), g, H)

# the following methods can be re-written to use analytic derivatives if desired
    def tan(self):
        return nonlinear_RVmath(lambda u: np.tan(u), [self])

    def acos(self):
        return nonlinear_RVmath(lambda u: np.acos(u), [self])

    def asin(self):
        return nonlinear_RVmath(lambda u: np.asin(u), [self])

    def atan(self):
        return nonlinear_RVmath(lambda u: np.atan(u), [self])

    def log(self):
        return nonlinear_RVmath(lambda u: np.log(u), [self])
        
    def __add__(self, other):
        """
        Override of __add__. Add two numbers together
        :param other: number to add to self
        :return: result
        """
        x = self._mean

        if isinstance(other, RandomVariable):
            y = other._mean
        else:
            y = other

        g = np.array([1.0, 1.0])
                      
        H = np.array([[0.0, 0.0],
                      [0.0, 0.0]])
        
        return _mean_var_from_derivs([self, other], x + y, g, H)

    __radd__ = __add__

    def __sub__(self, other):
        x = self._mean

        if isinstance(other, RandomVariable):
            y = other._mean
        else:
            y = other

        g = np.array([1.0, -1.0])
                      
        H = np.array([[0.0, 0.0],
                      [0.0, 0.0]])

        return _mean_var_from_derivs([self, other], x - y, g, H)

    def __rsub__(self, other):
        x = self._mean

        if isinstance(other, RandomVariable):
            y = other._mean
        else:
            y = other

        g = np.array([-1.0, 1.0])

        H = np.array([[0.0, 0.0],
                      [0.0, 0.0]])

        return _mean_var_from_derivs([self, other], y - x, g, H)

    def __mul__(self, other):
        x = self._mean

        if other is self:
            return self.__pow__(2.0)
        elif isinstance(other, RandomVariable):
            y = other._mean
        else:
            y = other

        g = np.array([y, x])
        H = np.array([[0.0, 1.0],
                      [1.0, 0.0]])

        return _mean_var_from_derivs([self, other], x*y, g, H)

    __rmul__ = __mul__

    def __div__(self, other):
        # x/y with y--other x--self
        x = self._mean

        if isinstance(other, RandomVariable):
            y = other._mean
        else:
            y = other

        g = np.array([1.0/y, -x/y**2.])
        H = np.array([[0.0, -1./y**2],
                      [-1./y**2, 2.*x/y**3]])

        return _mean_var_from_derivs([self, other], x/y, g, H)
    
    __truediv__ = __div__

    def __rdiv__(self, other):
        # x/y with y--self x--other
        y = self._mean

        if isinstance(other, RandomVariable):
            x = other._mean
        else:
            x = other

        g = np.array([1.0/y, -x/y**2.])

        H = np.array([[0.0, -1./y**2],
                      [-1./y**2, 2.*x/y**3]])

        return _mean_var_from_derivs([other, self], x/y, g, H)

    __rtruediv__ = __rdiv__

    def __pow__(self, other, modulo=None):
        if isinstance(other, (float, int)):
            return nonlinear_RVmath(lambda u: u[0]**other, [self])
        elif isinstance(other, RandomVariable):
            return nonlinear_RVmath(lambda u: u[0]**u[1], [self, other])
        else:
            raise NotImplementedError("Can only use pow with integers, floats, and Variables")

    def __rpow__(self, other):
        if isinstance(other, (float, int)):
            return nonlinear_RVmath(lambda u: other ** u[0], [self])
        elif isinstance(other, RandomVariable):
            return nonlinear_RVmath(lambda u: u[1] ** u[0], [self, other])
        else:
            raise NotImplementedError("Can only use pow with integers, floats, and Variables")

    def __neg__(self):
        return -1.0*self

    def __pos__(self):
        if self._mean > 0:
            return self
        else:
            return -self

    def __abs__(self):
        return self.__pos__()

    def __lt__(self, other):
        s = self._mean 
        if isinstance(other, RandomVariable):
            o = other._mean
        else:
            o = other

        return s < o

    def __le__(self, other):
        s = self._mean
        if isinstance(other, RandomVariable):
            o = other._mean
        else:
            o = other

        return s <= o

    def __eq__(self, other):
        s = self._mean
        if isinstance(other, RandomVariable):
            o = other._mean
        else:
            o = other

        return s == o

    def __ne__(self, other):
        s = self._mean
        if isinstance(other, RandomVariable):
            o = other._mean
        else:
            o = other

        return not s == o

    def __gt__(self, other):
        s = self._mean
        if isinstance(other, RandomVariable):
            o = other._mean
        else:
            o = other

        return s > o

    def __ge__(self, other):
        s = self._mean
        if isinstance(other, RandomVariable):
            o = other._mean
        else:
            o = other

        return s >= o
