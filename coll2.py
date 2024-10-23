import numpy as np
import sympy as sm
from sympy.abc import x, y

import matplotlib.pyplot as plt


ode = lambda u: (x**4)*u.diff(x).diff(x) + (x**6)*u.diff(x) - (x**5)*u - (6 - 3*x**3)


alpha0 = 1
alpha1 = 0
beta0 = 3
beta1 = 1

a = 1
b = 2
A = 1
B = 0.5

N = 3






def Collocation(alpha0, alpha1, beta0, beta1, a, b, A, B, N):
    """
    Collocation method for boundary-value problem
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
        c_i = sm.Symbol(f'c{i}')
        
        CONST.append(c_i)
        

    y_1 = 0
    for i in range(len(PHI)):
        y_1 += CONST[i]*PHI[i]


    ode2 = sm.lambdify([x], ode(y_1))

    step_ = (b - a)/N

    ODEs = []

    for i in range(1, N):
        ODEs.append(ode2(a + i*step_))


    COEFS = sm.solve(ODEs, CONST[1:], dict=True)[0]
    ANS = PHI[0]

    for i, c_i in enumerate(CONST[1:], 1):
        ANS += COEFS[c_i]*PHI[i]
        
    return sm.sympify(ANS)


if __name__ == "__main__":
    ANS_2 = sm.lambdify([x], Collocation(alpha0=alpha0, alpha1=alpha1, beta0=beta0, beta1=beta1, a=a, b=b, A=A, B=B, N=2))
    ANS_3 = sm.lambdify([x], Collocation(alpha0=alpha0, alpha1=alpha1, beta0=beta0, beta1=beta1, a=a, b=b, A=A, B=B, N=3))
    ANS_4 = sm.lambdify([x], Collocation(alpha0=alpha0, alpha1=alpha1, beta0=beta0, beta1=beta1, a=a, b=b, A=A, B=B, N=4))
    
    x = np.linspace(a, b, 1000)
    fig, ax = plt.subplots()
    ax.plot(x, [t**-2 for t in x], color='black', label='Correct answer')
    ax.plot(x, [ANS_2(t) for t in x], color='red', label='Collocation 2')
    ax.plot(x, [ANS_3(t) for t in x], color='green', label='Collocation 3')
    ax.plot(x, [ANS_4(t) for t in x], color='blue', label='Collocation 4')
    
    ax.legend(loc='upper left')
    plt.show()