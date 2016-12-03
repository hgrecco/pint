from pint.compat.stochastic_math import covariance_matrix
from pint import UnitRegistry
ureg = UnitRegistry()
Q_ = ureg.Quantity
M_ = ureg.Measurement

m1 = M_(2.0, 1.0, 'm')
m2 = M_(1.0, 2.0, 'ft')

prod = m1 * m2

covmat = covariance_matrix([prod, m1, m2])
cov_p_m1  = Q_(covmat[0, 1], m1.units*prod.units)
cov_p_m2  = Q_(covmat[0, 2], m2.units*prod.units)
cov_m1_m2 = Q_(covmat[1, 2], m1.units*m2.units)
print covmat
print("Cov(prod, m1): {}".format(cov_p_m1))
print("Cov(prod, m2): {}".format(cov_p_m2))
print("Cov(m2,   m1): {}".format(cov_m1_m2))

print("Base units")
print("Cov(prod, m1): {}".format(cov_p_m1.to_base_units()))
print("Cov(prod, m2): {}".format(cov_p_m2.to_base_units()))
print("Cov(m2,   m1): {}".format(cov_m1_m2.to_base_units()))

m3 = M_(2.0, 1.0, 'm')
v = (1.0 * ureg.ft).to('m').magnitude
s = (2.0 * ureg.ft).to('m').magnitude
m4 = M_(v, s, 'm')

prod2 = m3 * m4

covmat2 = covariance_matrix([prod2, m3, m4])
cov_p2_m3 = Q_(covmat2[0, 1], m3.units*prod2.units)
cov_p2_m4 = Q_(covmat2[0, 2], m4.units*prod2.units)
cov_m3_m4 = Q_(covmat2[1, 2], m3.units*m4.units)
print covmat
print("Cov(prod, m1): {}".format(cov_p2_m3))
print("Cov(prod, m2): {}".format(cov_p2_m4))
print("Cov(m2,   m1): {}".format(cov_m3_m4))
