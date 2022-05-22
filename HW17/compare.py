'''
HW 14-2: comparation between several constrained optimization algorithm
'''
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv, svd, lstsq
from numpy.random import seed, rand

class F:
    def __init__(self, A=None, b=None, n=500, p=100, random=True):
        if random:
            self.A = rand(p, n)
            self.b = rand(p)
        else:
            self.A = A
            self.b = b
        self.n = n
        self.p = p

    def f(self, x):
        return np.log(np.sum(np.exp(x)))

    def df(self, x):
        return np.exp(x) / np.sum(np.exp(x))
    
    def ddf(self, x):
        df = F.df(x)
        return np.diag(df) - np.outer(df, df)

class Opt:
    def __init__(self):
        pass

    def fe(self, z):
        return self.f(self.spec_sol + self.zero_space @ z)

    def dfe(self, z):
        return self.zero_space.transpose() @ self.df(self.spec_sol + self.zero_space @ z)

    def _lsearch(self, d, f, g):
        t = 1
        while self.f(self.x + t * d) > f + self.alpha * t * np.dot(d, g):
            t *= self.beta
        return t

    def _step(self, step, method):
        if method == "proj_grad":
            f = self.f(self.x)
            g = self.df(self.x)
            d = -self.P @ g
        elif method == "eliminate":
            if step == 0:
                self.x = np.zeros(self.n - self.p)
            f = self.fe(self.x)
            g = self.dfe(self.x)
            d = -g

        print(f)
        self.fs.append(f)
        if np.dot(d, d) < self.eta2:
            return False

        alpha = self._lsearch(d, f, g)
        self.x += alpha * d
        return True

    def opt(self, F, x0, alpha, beta, eta, method):
        self.f = F.f
        self.n = F.n
        self.p = F.p
        A = F.A
        self.A = A
        self.b = F.b
        self.P = np.eye(F.n) - A.transpose() @ inv(A @ A.transpose()) @ A
        self.fs = []
        self.df = F.df
        self.x = self.P @ x0 + A.transpose() @ (inv(A @ A.transpose()) @ F.b)
        self.alpha = alpha
        self.beta = beta
        self.eta2 = eta ** 2
        self.spec_sol = self.x.copy()
        u, s, vt = svd(self.A)
        self.zero_space = vt[self.p:].transpose()
        flag = True
        step = 0
        while flag:
            flag = self._step(step, method)
            step += 1
        self.L = len(self.fs)
        self.pstar = self.fs[-1]
        print(self.pstar)

    def plot_fi(self):
        plt.figure()
        plt.plot(range(self.L), np.log10(abs(np.array(self.fs) - self.pstar)))
        plt.show()

def test_F(method):
    seed(114514)
    opt = Opt()
    x0 = rand(500)
    func = F()
    opt.opt(F=func, x0=x0, alpha=0.5, beta=0.5, eta=1e-5, method=method)
    opt.plot_fi()

if __name__ == "__main__":
    test_F("eliminate")

    