import numpy as np
import sympy as sm
from sympy.abc import x, y

import matplotlib.pyplot as plt
import scipy.special as sc

a = 0
b = 1

c = 3

def ans(x):
    return 1/2 * (55 - 55*2.718281**(-x) - 13*x - 26*x**2 + 4*x**3 + 55*x*sc.expi(-1) - 55*x*sc.expi(-x) + 55*x*np.log(abs(-x)))


if __name__ == "__main__":

    # x = np.linspace(-c, c, 1000)
    x = np.linspace(a, b, 1000)
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title('Solution')
    plt.title('Solution')
    plt.ylim(-0.31, 0.02)
    plt.grid(1)
    
    plt.plot(x, ans(x), color='black')
    # plt.ylim(-2000, 15000)
    
    # ax.legend(loc='upper left')
    
    plt.savefig('img/Solution.png', dpi=600)
    plt.show()
