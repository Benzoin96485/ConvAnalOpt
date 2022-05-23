'''
HW 14-2: comparation between several constrained optimization algorithm
'''
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv, svd, solve
from numpy.random import seed, rand
from scipy.optimize import minimize
import random
random.seed(114514)
seed(114514)

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
        self.P = np.eye(n) - self.A.T @ inv(self.A @ self.A.T) @ self.A


    def f(self, x):
        return np.log(np.sum(np.exp(x)))

    def df(self, x):
        return np.exp(x) / np.sum(np.exp(x))
    
    def ddf(self, x):
        df = self.df(x)
        return np.diag(df) - np.outer(df, df)

    def proj(self, x):
        return self.P @ x + self.A.T @ (inv(self.A @ self.A.T) @ self.b)


class Opt:
    def __init__(self):
        pass

    def fe(self, z):
        return self.f(self.spec_sol + self.zero_space @ z)

    def dfe(self, z):
        return self.zero_space.T @ self.df(self.spec_sol + self.zero_space @ z)

    def _lsearch(self, x, d, f, g, func):
        t = 1
        while func(x + t * d) > f + self.alpha * t * np.dot(d, g):
            t *= self.beta
        return t

    def Lx(self, x):
        Axb = self.A @ x - self.b
        return self.f(x) + np.dot(self.v, Axb) + self.aug / 2 * np.dot(Axb, Axb)

    def dLx(self, x):
        return self.df(x) + self.A.T @ self.v + self.aug * self.A.T @ (self.A @ x - self.b)

    def _step(self, step, method):
        if method == "proj_grad":
            f = self.f(self.x)
            g = self.df(self.x)
            d = -self.P @ g
            l = d
        elif method == "eliminate":
            if step == 0:
                self.x = np.zeros(self.n - self.p)
            f = self.fe(self.x)
            g = self.dfe(self.x)
            d = -g
            l = d
        elif method == "newton":
            f = self.f(self.x)
            g = self.df(self.x)
            H = self.ddf(self.x)
            KKT_mat1 = np.concatenate([H, self.A.T], axis=1)
            KKT_mat2 = np.concatenate([self.A, np.zeros((self.p, self.p))], axis=1)
            KKT_mat = np.concatenate([KKT_mat1, KKT_mat2], axis=0)
            g0 = np.concatenate([-g, np.zeros(self.p)])
            d = solve(KKT_mat, g0)[:500]
            #l = np.dot(d, H @ d) / np.sqrt(2)
            l = d
        elif method == "dual_asc":
            f = self.f(self.x)
            L = self.Lx(self.x) 
            g = self.dLx(self.x)
            d = -g
            l = d
            pass # 啥时候能写完啊

        print(f)
        # print(np.max(self.A @ self.x - self.b))
        self.fs.append(f)
        # if np.dot(l, l) < self.eta2:
        if np.dot(l, l) < self.eta2 or step > 5000:
            return False

        if method == "dual_asc":
            if self.prec:
                self.x = minimize(self.Lx, self.x, jac=self.dLx).x
            else:
                alpha = self._lsearch(self.x, d, L, g, self.Lx)
                self.x += alpha * d
            self.v += self.aug * (self.A @ self.x - self.b)
        else:
            alpha = self._lsearch(self.x, d, f, g, self.f)
            self.x += alpha * d
        return True

    def opt(self, F, alpha, beta, eta, method, aug=0.000001, prec=False):
        self.x = F.proj(rand(F.n))
        self.v = np.zeros(F.p)
        self.f = F.f; self.n = F.n; self.p = F.p
        self.A = F.A; self.b = F.b; self.P = F.P
        self.df = F.df; self.ddf = F.ddf; self.aug = aug
        self.prec = prec
        u, s, vt = svd(self.A)
        self.zero_space = vt[self.p:].T
        self.fs = []
        self.alpha = alpha
        self.beta = beta
        self.eta2 = eta ** 2
        self.spec_sol = self.x.copy()
        
        flag = True
        step = 0
        while flag:
            flag = self._step(step, method)
            step += 1
        self.L = len(self.fs)
        # self.pstar = self.fs[-1]
        self.pstar = 6.215668419620234

    def plot_fi(self):
        plt.figure()
        plt.plot(np.linspace(1, self.L, self.L), np.log10(abs(np.array(self.fs) - self.pstar) + 1e-16))
        plt.show()

def test_F(A,b,method, **kwarg):
    opt = Opt()
    func = F(A,b,random=False)
    opt.opt(F=func, alpha=0.25, beta=0.75, eta=1e-5, method=method, **kwarg)
    ans = opt.fs[:]
    del opt
    return ans

if __name__ == "__main__":
    A = rand(100, 500)
    b = rand(100)
    l1 = test_F(A, b, method="proj_grad")
    l2 = test_F(A, b, method="eliminate")
    l3 = test_F(A, b, method="newton")
    l4 = test_F(A, b, method="dual_asc", aug=1e-6)
    l5 = test_F(A, b, method="dual_asc", aug=1e-3)
    pstar = 6.215668419620234
    plt.figure()
    plt.plot(range(len(l1)), np.log10(abs(np.array(l1) - pstar) + 1e-16), label="proj_grad", linewidth=3)
    plt.plot(range(len(l2)), np.log10(abs(np.array(l2) - pstar) + 1e-16), label="eliminate")
    plt.plot(range(len(l3)), np.log10(abs(np.array(l3) - pstar) + 1e-16), label="newton")
    plt.plot(range(len(l4)), np.log10(abs(np.array(l4) - pstar) + 1e-16), label="dual_asc, aug_L, beta=1e-6")
    plt.plot(range(len(l5)), np.log10(abs(np.array(l5) - pstar) + 1e-16), label="dual_asc, aug_L, beta=1e-3")
    plt.legend()
    plt.savefig("HW17/compare.png")
    plt.show()

"""
6.215668419622828
6.21566841962287
6.215668419620234
"""