import numpy as np
import sympy as sm
from sympy.abc import x, y

import scipy.special as sc

import matplotlib.pyplot as plt


ode = lambda u: (u.diff(x).diff(x) + u.diff(x) - (1/x)*u - (4*x**2 - x + 3/2))*sm.exp(x) # самосопряженное уравнение
px = sm.exp(x)
qx = sm.exp(x)/x
fx = sm.exp(x)*(4*x**2 - x + 3/2)

alpha0 = 1
alpha1 = 0
beta0 = 0
beta1 = 1

a = 0
b = 1
A = 0
B = 1

N = 2


def Ritz(alpha0, alpha1, beta0, beta1, a, b, A, B, N):
    """
    Ritz method for boundary-value problem
    """

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
        c_i = sm.Symbol(f'c{i+1}')
        CONST.append(c_i)
        
    # c1 = sm.Symbol('c1')
    # CONST.append(c1)
    print(CONST)


    y_1 = 0

    for i in range(N):
        y_1 += CONST[i]*PHI[i]
        
        
    y_1 = sm.simplify(y_1)
    print(y_1)


    RES = [None]*(N-1)

    for i in range(1, N):
        RES[i-1] = sm.integrate( ((px*y_1.diff(x)).diff(x) - qx*y_1 - fx)*PHI[i], (x, a, b))
        
    print(RES)
        
    COEF = sm.solve(RES, CONST, dict=True)[0]
    print(sm.solve(RES, CONST, dict=True))

    ANS = PHI[0]

    for i in range(1, N):
        ANS += COEF[CONST[i]]*PHI[i]
        
    return sm.simplify(ANS)