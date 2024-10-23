import numpy as np
import sympy as sm
from sympy.abc import x, y


ode = lambda u: (x**4)*u.diff(x).diff(x) + (x**6)*u.diff(x) - (x**5)*u - (6 - 3*x**3)


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
# print('0 -- ', phi0)

for i in range(1, N):
    pass


gamma1 = -(beta0*(b - a)**2 + (1 + 1)*beta1*(b - a))/(beta0*(b - a) + 1*beta1)
phi1 = (x - a)**2 + gamma1*(x - a)
# print('1 -- ', phi1)


PHI = [phi0, phi1]

CONST = [1]
c1 = sm.Symbol('c1')
c2 = sm.Symbol('c2')
CONST.append(c1)

y_ = 0
for i in range(len(PHI)):
    y_ += CONST[i]*PHI[i]

# print(y_.diff(x))
# print(ode(y_))
ode2 = sm.lambdify([x], ode(y_))
# print(ode2(1.5))


ode__ = ode2(1.5)

# print(ode__.coeff(c1))


print(sm.solve(ode__, CONST[1:], dict=True))