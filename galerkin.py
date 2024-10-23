import numpy as np
import sympy as sm
from sympy.abc import x, y

import matplotlib.pyplot as plt


ode = lambda u: (x**4)*u.diff(x).diff(x) + (x**6)*u.diff(x) - (x**5)*u - (6 - 3*x**3)
px = x**2
qx = -x
fx = (6 - 3*x**3)/(x**4)


alpha0 = 1
alpha1 = 0
beta0 = 3
beta1 = 1

a = 1
b = 2
A = 1
B = 0.5

N = 2



sys0_coef = np.array([
    [alpha0, alpha0*a + alpha1],
    [beta0, beta0*b + beta1]
])

sys0_right = np.array([
    A,
    B
])

phi0_coef = np.linalg.solve(sys0_coef, sys0_right)

phi0 = phi0_coef[0] + phi0_coef[1]*x

PHI = [phi0]

for i in range(1, N):
    gamma_i = -(beta0*(b - a)**2 + (i + 1)*beta1*(b - a))/(beta0*(b - a) + i*beta1)
    phi_i = gamma_i*(x - a)**i + (x - a)**(i + 1)
    
    PHI.append(phi_i)


CONST = [1]

for i in range(1, N):
    c_i = sm.Symbol(f'c{i}')
    
    CONST.append(c_i)
    

y_1 = 0
for i in range(len(PHI)):
    y_1 += CONST[i]*PHI[i]


ode2 = sm.lambdify([x], ode(y_1))



a11 = sm.simplify(
    sm.lambdify([x], PHI[1]*PHI[1].diff(x))(b) - sm.lambdify([x], PHI[1]*PHI[1].diff(x))(a) - \
    sm.integrate(PHI[1].diff(x)*PHI[1].diff(x), (x, a, b)) + \
    sm.integrate(px*PHI[1].diff(x)*PHI[1], (x, a, b)) + \
    sm.integrate(qx*PHI[1]*PHI[1], (x, a, b))
)

print(a11)

d1 = sm.simplify(
    sm.integrate((fx - PHI[0].diff(x).diff(x) - px*PHI[0].diff(x) - qx*PHI[0])*PHI[1], (x, a, b))
)

print(d1)


print(sm.simplify(d1/a11))








# step_ = (b - a)/N

# ODEs = []

# for i in range(1, N):
#     ODEs.append(ode2(a + i*step_))


# COEFS = sm.solve(ODEs, CONST[1:], dict=True)[0]
# ANS = PHI[0]

# for i, c_i in enumerate(CONST[1:], 1):
#     ANS += COEFS[c_i]*PHI[i]
    
# ANS = sm.sympify(ANS)