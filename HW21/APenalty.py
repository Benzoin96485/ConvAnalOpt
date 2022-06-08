'''
HW 21-2: Accelerated penalty method for constrained log-sum-exp problem
'''
import numpy as np
import matplotlib.pyplot as plt
from time import time
from numpy.random import seed, randn, rand
from numpy.linalg import norm, inv, svd
import random

random.seed(114514)
seed(114514)

class Opt():
    def __init__(self, A, b):
        self.A = A
        self.b = b
        self.nb = norm(self.b)
        self.L = 3
        self.T = self.L
        self.n = self.A.shape[1]
        self.p = self.A.shape[0]
        self.eta = norm(A, ord=2) ** 2 + 2

    def f(self, x):
        return np.log(np.sum(np.exp(x)))

    def df(self, x):
        return np.exp(x) / np.sum(np.exp(x))

    # def fe(self, z):
    #     return self.f(self.spec_sol + self.zero_space @ z)

    # def dfe(self, z):
    #     return self.zero_space.T @ self.df(self.spec_sol + self.zero_space @ z)

    def _step_AP(self, step):
        if step == 0:
            self.ATA = self.A.T @ self.A
            self.ATb = self.A.T @ self.b
        alpha = 1 / (step + 1)
        theta = 1 / (step + 1)
        eta = self.L + self.beta * norm(self.ATA, ord=2) / alpha
        x_ = self.x + (eta * theta - self.mu) * (1 - self.theta) / (eta - self.mu) / self.theta * self.dx
        x = x_ - (self.df(x_) + self.beta / alpha * (self.ATA @ x_ - self.ATb)) / eta
        self.theta = theta
        self.dx = x - self.x
        self.x = x
        self.Ax_b = self.A @ self.x - self.b
        self.con = norm(self.Ax_b)
        return True

    # def _step_eli(self, step):
        
    #     def fe(z):
    #         return self.f(self.spec_sol + self.zero_space @ z)

    #     def dfe(z):
    #         return self.zero_space.T @ self.df(self.spec_sol + self.zero_space @ z)

    #     def _lsearch(x, d, f, g, func):
    #         t = 1
    #         while func(x + t * d) > f + 0.25 * t * np.dot(d, g):
    #             t *= 0.75
    #         return t

    #     if step == 0:
    #         self.P = np.eye(self.n) - self.A.T @ inv(self.A @ self.A.T) @ self.A
    #         self.x = np.zeros(self.n - self.p)
    #         u, s, vt = svd(self.A)
    #         self.zero_space = vt[self.p:].T
    #         self.spec_sol = self.A.T @ (inv(self.A @ self.A.T) @ self.b)

    #     self.fstar = fe(self.x)
    #     g = dfe(self.x)
    #     d = -g
    #     l = d
    #     if np.dot(l, l) < 1e-16:
    #         return False
    #     alpha = _lsearch(self.x, d, self.fstar, g, fe)
    #     self.x += alpha * d
    #     return True

    def _step_prox(self, step):
        flag1, flag2 = True, True
        self.tau = self.T + self.beta * self.eta
        x = self.x - (self.A.T @ self.lamda + self.df(self.x)) / self.tau
        self.Ax_b = self.A @ x - self.b
        self.con = norm(self.Ax_b)
        self.lamda += self.beta * self.Ax_b
        if self.con / self.nb < self.eps1:
            flag1 = False
        if np.sqrt(self.eta) * norm(x - self.x) * self.beta / self.nb < self.eps2:
            self.beta = min(self.rho * self.beta, self.beta_max)
            flag2 = False
        self.x = x.copy()
        return flag1 or flag2
            
    def opt(self, x0, method, beta, lamda=None, rho=1.01, beta_max=1e4, eps1=1e-5, eps2=1e-3):
        self.x = x0
        self.Ax_b = self.A @ self.x - self.b
        self.con = norm(self.Ax_b)
        self.dx = np.zeros_like(x0)
        self.method = method
        self.theta = 1
        self.beta = beta
        self.beta_max = beta_max
        self.lamda = lamda
        self.rho = rho
        self.mu = 0
        self.fs = []
        self.ts = []
        self.cs = []
        self.eps1 = eps1
        self.eps2 = eps2
        self.pstar = 6.2156684196202585
        self._step = getattr(self, f"_step_{method}")
        flag = True
        step = 0
        self.t0 = time()
        while flag and (not self.ts or self.ts[-1] < 10):
            self.fstar = self.f(self.x)
            self.ts.append(time() - self.t0)
            self.cs.append(self.con)
            flag = self._step(step)
            self.fs.append(self.fstar)
            step += 1
            if step % 1 == 0:
                print(self.fstar)
    
    def plot_ft(self): 
        plt.plot(self.ts, np.log10(np.abs(np.array(self.fs) - self.pstar)), label=f"{self.method}")

def test_F(A, b, x0, method, beta, lamda=None, rho=1.01):
    opt = Opt(A.copy(), b.copy())
    opt.opt(x0, method, beta, lamda)
    opt.plot_ft()
    ts, cons = opt.ts.copy(), opt.cs.copy()
    del opt
    return ts, cons

if __name__ == "__main__":
    A = rand(100, 500)
    b = rand(100)
    x0 = np.zeros(500)
    plt.figure()
    ts1, cons1 = test_F(A, b, x0, "prox", 0.000001, np.zeros(100), 1.0001)
    ts2, cons2 = test_F(A, b, x0, "AP", 0.000001, np.zeros(100), 1.0001)
    plt.legend()
    plt.xlabel("t/s")
    plt.ylabel(r"$\lg|f(x)-p^*|$")
    plt.show()
    plt.figure()
    plt.plot(ts1, np.log10(np.array(cons1)), label="prox")
    plt.plot(ts2, np.log10(np.array(cons2)), label="AP")
    plt.legend()
    plt.xlabel("t/s")
    plt.ylabel(r"$\lg\|\mathbf{A}x-b\|_{F}^{2}$")
    plt.show()