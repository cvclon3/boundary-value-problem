import numpy as np
import sympy as sm
from sympy.abc import x, y

import matplotlib.pyplot as plt
import scipy.special as sc

import pandas as pd
import matplotx


ode = lambda u: u.diff(x).diff(x) + u.diff(x) - (1/x)*u - (4*x**2 - x + 3/2)
px = 1
qx = -1/x
fx = 4*x**2 - x + 3/2


alpha0 = 1
alpha1 = 0
beta0 = 0
beta1 = 1

a = 0
b = 1
A = 0
B = 1

N = 2


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



def ans(x):
    return 1/2 * (55 - 55*2.718281**(-x) - 13*x - 26*x**2 + 4*x**3 + 55*x*sc.expi(-1) - 55*x*sc.expi(-x) + 55*x*np.log(abs(-x)))


def test(FUNC, N, n_iter=10):
    from datetime import datetime
    
    start =  datetime.now()
    
    for i in range(n_iter):
        FUNC(alpha0=alpha0, alpha1=alpha1, beta0=beta0, beta1=beta1, a=a, b=b, A=A, B=B, N=N)
        
    end = datetime.now()
    
    return end - start


if __name__ == "__main__":
    ANS_2 = sm.lambdify([x], Collocation(alpha0=alpha0, alpha1=alpha1, beta0=beta0, beta1=beta1, a=a, b=b, A=A, B=B, N=2))
    ANS_3 = sm.lambdify([x], Collocation(alpha0=alpha0, alpha1=alpha1, beta0=beta0, beta1=beta1, a=a, b=b, A=A, B=B, N=3))
    ANS_4 = sm.lambdify([x], Collocation(alpha0=alpha0, alpha1=alpha1, beta0=beta0, beta1=beta1, a=a, b=b, A=A, B=B, N=4))
    
    
    # print(f'N = 2: {test(Collocation, 2).microseconds/1000}')
    # print(f'N = 3: {test(Collocation, 3).microseconds/1000}')
    # print(f'N = 4: {test(Collocation, 10).microseconds/1000}')
    
    
    x = np.linspace(a, b, 1000)
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title('Collocation')
    plt.title('Collocation')
    plt.ylim(-0.31, 0.02)
    plt.grid(1)
    
    ax.plot(x, [ans(t) for t in x], color='black', label='1')
    ax.plot(x, [ANS_2(t) for t in x], color='red', label='2')
    ax.plot(x, [ANS_3(t) for t in x], color='green', label='3')
    ax.plot(x, [ANS_4(t) for t in x], color='blue', label='4')
    
    matplotx.line_labels()
    
    ax.legend(loc='lower left', labels=[
        '1 - Correct answer', 
        '2 - Collocation 2', 
        '3 - Collocation 3', 
        '4 - Collocation 4'
    ])
    
    plt.savefig('img/Collocation2.png', dpi=600)
    plt.show()
    
    # DATA
    x_val = np.linspace(a, b, 10)
    y_val = ans(x_val)
    y_2 = ANS_2(x_val)
    y_3 = ANS_3(x_val)
    y_4 = ANS_4(x_val)
    
    df = pd.DataFrame({
        'x_i': x_val,
        'y_i': y_val,
        'n = 2': y_2, 
        'delta2': y_val-y_2,
        'n = 3': y_3, 
        'delta3': y_val-y_3, 
        'n = 4': y_4,
        'delta4': y_val-y_4,
    })
    
    print(df)
    df.to_excel('data/Collocation.xlsx')
    df.to_csv('data/Collocation.csv')
