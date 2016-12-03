import numpy as np
from pint import UnitRegistry
ureg = UnitRegistry()
Q_ = ureg.Quantity
M_ = ureg.Measurement

q = 2.0 * ureg.meter
m = (1.0 * ureg.meter).plus_minus(0.1)

m.value

# -----------------checking addition---------------------------------------
r = q + m
print "q + m ==> " + str(type(r)) + " (should be measurement)"
print str(r) + '\n'

r = m + q
print "m + q ==> " + str(type(r)) + " (should be measurement)"
print str(r) + '\n'

r = q + q
print "q + q ==> " + str(type(r)) + " (should be quantity)"
print str(r) + '\n'

r = m + m
print "m + m ==> " + str(type(r)) + " (should be measurement)"
print str(r) + '\n'


# ------------------checking subtraction-------------------------------------
r = q - m
print "q - m ==> " + str(type(r)) + " (should be measurement)"
print str(r) + '\n'

r = m - q
print "m - q ==> " + str(type(r)) + " (should be measurement)"
print str(r) + '\n'

r = q - q
print "q - q ==> " + str(type(r)) + " (should be quantity)"
print str(r) + '\n'

r = m - m
print "m - m ==> " + str(type(r)) + " (should be measurement)"
print str(r) + '\n'

# ------------------checking multiplication-------------------------------------
r = q * m
print "q * m ==> " + str(type(r)) + " (should be measurement)"
print str(r) + '\n'

r = m * q
print "m * q ==> " + str(type(r)) + " (should be measurement)"
print str(r) + '\n'

r = q * q
print "q * q ==> " + str(type(r)) + " (should be quantity)"
print str(r) + '\n'

r = m * m
print "m * m ==> " + str(type(r)) + " (should be measurement)"
print str(r) + '\n'

# ------------------checking division-------------------------------------
r = q / m
print "q / m ==> " + str(type(r)) + " (should be measurement)"
print str(r) + '\n'

r = m / q
print "m / q ==> " + str(type(r)) + " (should be measurement)"
print str(r) + '\n'

r = q / q
print "q / q ==> " + str(type(r)) + " (should be quantity)"
print str(r) + '\n'

r = m / m
print "m / m ==> " + str(type(r)) + " (should be measurement)"
print str(r) + '\n'

m_angle = M_(1.0, 1.0, 'radians')
r = np.sin(m_angle)
print "sin(m) ==> " + str(type(r)) + " (should be measurement)"
print str(r) + '\n'

m2 = M_(2.0, 1.0, 'feet')
r = m2/m
print "m2/m ==> " + str(type(r)) + " (should be measurement)"
print str(r) + '\n'

r = (m2-m)/m
print "(m2-m)/m ==> " + str(type(r)) + " (should be measurement)"
print r

# try:
#     r = np.log(m2)
#     print 'log(m2)'
#     print r
# except:
#     print "taking log of dimensioned quantity failed"

r = np.log(q/q)
print "log(q)==> " + str(type(r)) + " (should be measurement)"
print r

r1 = m2/m
r = np.log(r1)
print "log(m2/m)==> " + str(type(r)) + " (should be measurement)"
print r