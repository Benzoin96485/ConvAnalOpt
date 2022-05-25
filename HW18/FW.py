'''
HW 18-2: compare Frank-Wolfe algorithm and projection gradient descent method on least-square problem with norm constraint
'''
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import seed, randn
from scipy.optimize import minimize

import random
random.seed(114515)
seed(114515)

class F:
    def __init__(self, D, y, norm):
        self.D = D
        self.y = y
        self.norm = norm

    def proj(self, x):
        if self.norm == "l1":
            x1 = x.copy()
            dn = np.linalg.norm(x1, 1) - 1
            while dn > 1e-14:
                x1 = x1 - dn * np.sign(x1) / np.linalg.norm(np.sign(x1), 1)
                dn = np.linalg.norm(x1, 1) - 1
            return x1
        elif self.norm == "linf":
            x1 = np.minimum(np.ones_like(x), x)
            x2 = np.maximum(-np.ones_like(x1), x1)
            return x2
    
    def f(self, x):
        yDx = self.y - self.D @ x
        return np.dot(yDx, yDx)
    
    def df(self, x):
        return 2 * self.D.T @ (self.D @ x - self.y)

class Opt:
    def __init__(self):
        pass

    def _lsearch(self, x, d, f, g, func):
        t = 1
        while func(x + t * d) > f + self.alpha * t * np.dot(d, g):
            t *= self.beta
        return t

    def _step(self, step, method):
        f = self.f(self.x)
        print(f)
        self.fs.append(f)
        g = self.df(self.x)
        x = self.x.copy()
        if method == "proj":
            d = -g
            self.fs.append(f)
            alpha = self._lsearch(self.x, d, f, g, self.f)
            self.x += alpha * d
            self.x = self.proj(self.x)
        elif method == "FW":
            gamma = 2 / (step + 2)
            if self.norm == "l1":
                index = np.argmax(np.abs(g))
                self.x = (1 - gamma) * self.x
                self.x[index] -= gamma * np.sign(g[index])
            elif self.norm == "linf":
                d = -np.sign(g)
                self.x += gamma * (d - self.x)
        dx = self.x - x
        if np.dot(dx, dx) < self.eta2 or step > 5000:
            return False
        return True

    def opt(self, F, x0, alpha, beta, eta, method):
        self.D = F.D; self.y = F.y
        self.x = x0; self.proj = F.proj
        self.f = F.f; self.df = F.df; self.norm = F.norm
        self.alpha = alpha; self.beta = beta; self.eta2 = eta ** 2
        self.fs = []
        flag = True
        step = 0
        while flag:
            flag = self._step(step, method)
            step += 1 
        self.L = len(self.fs)

    def plot_fi(self):
        plt.figure()
        plt.plot(np.linspace(1, self.L, self.L), np.log10(abs(np.array(self.fs) - self.pstar) + 1e-16))
        plt.show()

def test_F(D, y, x0, norm, method):
    opt = Opt()
    func = F(D, y, norm)
    opt.opt(F=func, x0=x0, alpha=0.25, beta=0.75, eta=1e-5, method=method)
    ans = opt.fs[:]
    del opt
    return ans

if __name__ == "__main__":
    D = randn(200, 300)
    y = randn(200)
    x0 = randn(300)
    func = F(D, y, "l1")
    cons1 = ({'type': 'ineq', 'fun': lambda x: 1 - np.linalg.norm(x, 1)})
    res = minimize(func.f, x0, constraints=cons1, tol=1e-4, options={'maxiter': 1e4, 'disp': True})
    minValue = func.f(res.x)
    print("Scipy result for l1 norm:", minValue)
    pstar = minValue
    ans1 = test_F(D, y, x0, "l1", "proj")
    ans2 = test_F(D, y, x0, "l1", "FW")
    plt.figure()
    plt.plot(range(len(ans1)), np.log10(abs(np.array(ans1) - pstar) + 1e-16), label="l1: proj")
    plt.plot(range(len(ans2)), np.log10(abs(np.array(ans2) - pstar) + 1e-16), label="l1: FW")
    plt.legend()
    plt.savefig("l1.png")
    func = F(D, y, "linf")
    cons1 = ({'type': 'ineq', 'fun': lambda x: 1 - np.linalg.norm(x, float("inf"))})
    res = minimize(func.f, x0, constraints=cons1, tol=1e-4, options={'maxiter': 1e4, 'disp': True})
    minValue = func.f(res.x)
    print("Scipy result for linf norm:", minValue)
    pstar = minValue
    ans1 = test_F(D, y, x0, "linf", "proj")
    ans2 = test_F(D, y, x0, "linf", "FW")
    plt.figure()
    plt.plot(range(len(ans1)), np.log10(abs(np.array(ans1) - pstar) + 1e-16), label="linf: proj")
    plt.plot(range(len(ans2)), np.log10(abs(np.array(ans2) - pstar) + 1e-16), label="linf: FW")
    plt.legend()
    plt.savefig("linf.png")