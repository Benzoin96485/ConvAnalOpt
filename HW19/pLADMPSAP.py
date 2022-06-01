'''
HW 19-1: Proximal Linearized Alternating Direction Method with Parallel Splitting and Adaptive Penalty for Logistic regression problem
'''
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import seed, randn, randint
from numpy.linalg import norm

import random
random.seed(114515)
seed(114515)

class F:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.s = y.shape[0]
        self.T0 = 2
        self.T = norm(X, ord=2, axis=0) / self.s + 2
        self.eta0 = self.s ** 2 * (self.s + 2) + 2
        self.eta = (self.s + 2) * np.array([1] * self.s) + 2

    def f(self, w):
        return np.sum(np.log(1 + np.exp(-(w @ self.X) * self.y))) / self.s

    def df(self, w):
        z = -(w.T @ self.X) * self.y
        g = -self.y * (1 - 1 / (np.exp(z) + 1))
        return self.X @ g / self.s

class Opt:
    def __init__(self):
        pass

    def _lsearch(self, x, d, f, g, func):
        t = 100
        while func(x + t * d) > f + self.alpha * t * np.dot(d, g):
            t *= self.beta
        return t

    def _step(self):
        f = self.f(self.x)
        self.fs.append(self.f(self.x))
        if self.method == "grad":
            g = self.df(self.x)
            d = -g
            if np.dot(d, d) < self.eta2:
                return False 
            else:
                print(f"{f} eta2={np.dot(d, d)}")
            alpha = self._lsearch(self.x, d, f, g, self.f)
            self.x += alpha * d
            return True
        elif self.method == "pLADMPSAP":
            self.tau0 = self.T0 + self.bbeta * self.eta0
            self.tau = self.T + self.bbeta * self.eta

            dx = np.sum(self.Lamda, axis=1) / self.tau0
            dL = np.zeros_like(self.Lamda)
            for i in range(self.s):
                self.W[:,i] -= (self.Lamda[:,i] + self.df(self.W[:,i])) / self.tau[i]
                dL[:,i] = self.W[:,i] - self.x
            # dW = -(self.Lamda + self.df(self.W)) / self.tau.reshape(1, -1)
            self.x += dx
            # self.W += dW
            # dL = self.W - self.x.reshape(-1, 1)
            if norm(dL) ** 2 < self.eta2:
                return False
            else:
                print(f"{f} eta2={norm(dL) ** 2}")
            # eps0 = self.beta * np.max((self.eta0 * norm(dx), np.max(self.eta * norm(dW, ord=2, axis=0)))) 
            # if eps0 < self.eps:
            #     self.beta *= 2
            self.Lamda += self.bbeta * dL
            return True
    
    def opt(self, w0, F, method, alpha=0.25, beta=0.75, eta=1e-4, 
        W0=None, Lamda0=None, bbeta0=1e-3, eps=1e-3
    ):
        self.x = w0
        self.f = F.f
        self.df = F.df
        self.s = F.s
        self.eta = F.eta
        self.method = method
        self.alpha = alpha
        self.beta = beta
        self.eta2 = eta ** 2
        self.fs = []
        self.Lamda = Lamda0
        self.W = W0
        self.bbeta = bbeta0
        self.T0 = F.T0
        self.T = F.T
        self.eta0 = F.eta0
        self.eta = F.eta
        self.eps = eps
        flag = True
        step = 0
        while flag and step < 10000:
            flag = self._step()
            step += 1 
        self.L = len(self.fs)

    def plot_fi(self, s):
        plt.figure()
        self.pstar = 0.339642157042786
        plt.plot(np.linspace(1, self.L, self.L), np.log10(abs(np.array(self.fs) - self.pstar) + 1e-16))
        plt.ylabel(r"$\log(|f(x)-p^*|)$")
        plt.xlabel("iter")
        plt.savefig(f"19_1_{s}_fi.png")
        plt.show()

def test_F_grad(X, y, w0):
    opt = Opt()
    func = F(X, y)
    opt.opt(w0, func, method="grad", alpha=0.25, beta=0.75, eta=1e-6)
    ans = opt.fs[:]
    print(ans[-1])
    opt.plot_fi("grad")
    del opt
    return ans

def test_F_ADM(X, y, w0, W0, Lamda0, bbeta0):
    opt = Opt()
    func = F(X, y)
    opt.opt(w0, func, method="pLADMPSAP", W0=W0, Lamda0=Lamda0, bbeta0=bbeta0)
    ans = opt.fs[:]
    print(ans[-1])
    opt.plot_fi("adm")
    del opt
    return ans

if __name__ == "__main__":
    X = randn(100, 200)
    y = randint(0, high=2, size=200)
    w0 = np.zeros(100)
    W0 = np.zeros((100, 200))
    Lamda0 = np.zeros((100, 200))
    bbeta0 = 0.001
    # test_F_grad(X, y, w0)
    test_F_ADM(X, y, w0, W0, Lamda0, bbeta0)