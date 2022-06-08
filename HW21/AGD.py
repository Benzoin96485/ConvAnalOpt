'''
HW 21-1: Several accelerated method:
    - Accelerated gradient descent
    - AGD with backtracking
    - Monotone AGD
for logistic regression problem
'''

import numpy as np
import matplotlib.pyplot as plt
from time import time
from numpy.random import seed, randn, randint
from numpy.linalg import norm
import random
random.seed(114514)
seed(114514)

class Opt():
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.N = X.shape[1]
        self.M = X.shape[0]
        self.beta = norm(X, ord=2) ** 2 / 4 / self.N

    def f(self, w):
        return np.sum(np.log(1 + np.exp(-(w @ self.X) * self.y))) / self.N

    def df(self, w):
        z = -(w.T @ self.X) * self.y
        g = -self.y * (1 - 1 / (np.exp(z) + 1))
        return self.X @ g / self.N

    def _lsearch(self, x, d, f, g, func):
        t = 100
        while func(x + t * d) > f + self.alpha_ls * t * np.dot(d, g):
            t *= self.beta_ls
        return t

    def _lsearch_AGD(self, y, fy, dfy):
        Q = lambda beta, x, y, fy, dfy: fy + np.dot(x - y, dfy) + np.dot(x - y, x - y) * beta / 2
        p = lambda beta, y, dfy: y - dfy / beta
        while Q(self.L, p(self.L, y, dfy), y, fy, dfy) < self.f(p(self.L, y, dfy)):
            self.L *= self.eta

    def _step_grad(self, step):
        g = self.df(self.x)
        d = -g
        if np.dot(d, d) < self.eps2:
            return False
        alpha = self._lsearch(self.x, d, self.fstar, g, self.f)
        self.x += alpha * d
        return True

    def _step_AGD(self, step):
        dfx = self.df(self.x)
        if np.dot(dfx, dfx) < self.eps2:
            return False
        x_ = self.x - self.df(self.x) / self.beta
        lamda = (1 + np.sqrt(1 + 4 * self.lamda ** 2)) / 2
        gamma = (1 - self.lamda) / lamda
        self.x = (1 - gamma) * x_ + gamma * self.x_
        self.lamda = lamda
        self.x_ = x_
        return True

    def _step_AGD_bktk(self, step):
        fx_ = self.f(self.x_)
        dfx_ = self.df(self.x_)
        if np.dot(dfx_, dfx_) < self.eps2:
            return False
        self._lsearch_AGD(self.x_, fx_, dfx_)
        x = self.x_ - dfx_ / self.beta
        lamda = (1 + np.sqrt(1 + 4 * self.lamda ** 2)) / 2
        self.x_ = x + (self.lamda - 1) / lamda * (x - self.x)
        self.lamda = lamda
        self.x = x
        return True

    def _step_AGD_mono(self, step):
        self.x_ = self.x + self.lamdar * (self.x__ - self.x) + (self.lamdar - 1 / self.lamda) * self.dx
        dfx_ = self.df(self.x_)
        if np.dot(dfx_, dfx_) < self.eps2:
            return False
        self.x__ = self.x_ - dfx_ / self.beta
        lamda = (1 + np.sqrt(1 + 4 * self.lamda ** 2)) / 2
        self.lamdar = self.lamda / lamda
        self.lamda = lamda
        if self.f(self.x__) <= self.f(self.x):
            x = self.x__
        else:
            x = self.x
        self.dx = x - self.x
        self.x = x
        return True

    def opt(self, w0, method, alpha_ls=0.25, beta_ls = 0.75, eps=1e-5, eta=1.5):
        self.alpha_ls = alpha_ls
        self.beta_ls = beta_ls
        self.method = method
        self.lamda = 0 if method == "AGD" else 1
        self.L = self.beta
        self.x = w0
        self.x_ = w0
        self.x__ = w0
        self.dx = np.zeros_like(self.x)
        self.lamdar = 1
        self.eps2 = eps ** 2
        self.pstar = 0.6397146685539034
        self.fs = []
        self.ts = []
        self._step = getattr(self, f"_step_{method}")
        flag = True
        step = 0
        self.t0 = time()
        while flag:
            self.fstar = self.f(self.x)
            self.fs.append(self.fstar)
            self.ts.append(time() - self.t0)
            flag = self._step(step)
            step += 1
            # if step % 1 == 0:
            #     print(self.fstar)
        self.L = len(self.fs)

    def plot_ft(self): 
        plt.plot(self.ts, np.log10(np.abs(np.array(self.fs) - self.pstar)), label=f"{self.method}")

def test_F(X, y, w0, method=None):
    opt = Opt(X.copy(), y.copy())
    opt.opt(w0.copy(), method)
    opt.plot_ft()
    del opt

if __name__ == "__main__":
    X = randn(1000, 10000)
    y = randint(0, 2, size=10000) * 2 - 1
    w0 = np.zeros(1000)
    plt.figure(dpi=300)
    test_F(X, y, w0, "grad")
    test_F(X, y, w0, "AGD")
    test_F(X, y, w0, "AGD_bktk")
    test_F(X, y, w0, "AGD_mono")
    plt.xlabel("t/s")
    plt.ylabel(r"$\lg|f(x)-p^*|$")
    plt.legend()
    plt.savefig("HW21/1.png")
    plt.show()

