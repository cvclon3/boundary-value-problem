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


def Galerkin(alpha0, alpha1, beta0, beta1, a, b, A, B, N):
    """
    Galerkin method for boundary-value problem
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


    A_MTX = np.zeros((N-1, N-1))
    D_MTX = np.zeros(N-1)

    for i in range(1, N):
        for j in range(1, N):
            A_MTX[i-1][j-1] = sm.simplify(
                                sm.lambdify([x], PHI[i]*PHI[j].diff(x))(b) - sm.lambdify([x], PHI[i]*PHI[j].diff(x))(a) - \
                                sm.integrate(PHI[j].diff(x)*PHI[i].diff(x), (x, a, b)) + \
                                sm.integrate(px*PHI[j].diff(x)*PHI[i], (x, a, b)) + \
                                sm.integrate(qx*PHI[j]*PHI[i], (x, a, b))
                            )
            
            D_MTX[i-1] = sm.simplify(
                                sm.integrate((fx - PHI[0].diff(x).diff(x) - px*PHI[0].diff(x) - qx*PHI[0])*PHI[i], (x, a, b))
                            )


    COEFS = np.linalg.solve(A_MTX, D_MTX)

    print(A_MTX)
    print(D_MTX)

    print(COEFS)

    ANS = PHI[0]

    for i in range(1, N):
        ANS += COEFS[i-1]*PHI[i]
        
    return sm.simplify(ANS)


if __name__ == "__main__":
    ANS_2 = sm.lambdify([x], Galerkin(alpha0=alpha0, alpha1=alpha1, beta0=beta0, beta1=beta1, a=a, b=b, A=A, B=B, N=2))
    ANS_3 = sm.lambdify([x], Galerkin(alpha0=alpha0, alpha1=alpha1, beta0=beta0, beta1=beta1, a=a, b=b, A=A, B=B, N=3))
    ANS_4 = sm.lambdify([x], Galerkin(alpha0=alpha0, alpha1=alpha1, beta0=beta0, beta1=beta1, a=a, b=b, A=A, B=B, N=4))
    
    x = np.linspace(a, b, 1000)
    fig, ax = plt.subplots()
    ax.plot(x, [t**-2 for t in x], color='black', label='Correct answer')
    ax.plot(x, [ANS_2(t) for t in x], color='red', label='Collocation 2')
    ax.plot(x, [ANS_3(t) for t in x], color='green', label='Collocation 3')
    ax.plot(x, [ANS_4(t) for t in x], color='blue', label='Collocation 4')
    
    ax.legend(loc='upper left')
    plt.show()