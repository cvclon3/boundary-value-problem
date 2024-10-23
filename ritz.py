import numpy as np
import sympy as sm
from sympy.abc import x, y

import matplotlib.pyplot as plt


ode = lambda u: u.diff(x).diff(x) + u  - (-x)
px = 1
qx = -1
fx = -x


a = 0
b = 1
A = 0
B = 0

N = 1

PHI = []

PHI_1 = x*(b - x)*x**0
print(PHI_1)

PHI.append(PHI_1)


CONST = []
c1 = sm.Symbol('c1')
CONST.append(c1)


y_1 = 0

for i in range(N):
    y_1 += CONST[i]*PHI[i]
    

res = sm.integrate(((px*y_1.diff(x)).diff(x) - qx*y_1 - fx)*PHI[0], (x, a, b))


print(sm.solve(res, CONST, dict=True))

