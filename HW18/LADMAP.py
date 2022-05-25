'''
HW 18-3: Linearized Alternating Direction Method with Adaptive Penalty for graph construction problem
'''

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import seed, randn
from numpy.linalg import norm, svd

import random
random.seed(114515)
seed(114515)

class F:
    def __init__(self, D, mu):
        self.D = D
        self.lamda = mu
        self.A1 = np.vstack((self.D, np.ones(D.shape[1])))
        self.A2 = np.vstack((np.eye(D.shape[0]), np.zeros(D.shape[0])))
        self.b = np.vstack((self.D, np.ones(D.shape[1])))
        self.eta1 = norm(self.A1, 2) ** 2 + 1
        self.eta2 = norm(self.A2, 2) ** 2 + 1

    def f1(self, Z):
        return norm(Z, ord="nuc")

    def f2(self, E):
        return self.lamda * np.sum(norm(E, ord=2, axis=0))

    def f1_prox(self, W, beta):
        eps = 1 / beta
        U, Sigma, VT = svd(W)
        Sigma = np.maximum(Sigma - eps, np.zeros_like(Sigma))
        return U @ np.diag(Sigma) @ VT

    def f2_prox(self, W, beta):
        eps = 1 / beta
        norm2vec = norm(W, ord=2, axis=0)
        E = np.zeros_like(W)
        for i in range(E.shape[1]):
            if norm2vec[i] > eps:
                E[:, i] = (1 - eps / norm2vec[i]) * W[:, i]
        return E

class Opt:
    def __init__(self):
        pass

    def _step(self):
        flag1, flag2 = True, True
        f = self.f1(self.Z) + self.f2(self.E)
        self.fs.append(f)
        self.bs.append(self.beta)
        self.gap.append(norm(self.D - self.D @ self.Z + self.E))
        # update Z
        W1 = self.Z - self.A1.T @ (self.lamda + self.beta * (self.A1 @ self.Z + self.A2 @ self.E - self.b)) / (self.beta * self.eta1)
        Z = self.f1_prox(W1, self.beta * self.eta1)

        # update E
        W2 = self.E - self.A2.T @ (self.lamda + self.beta * (self.A1 @ Z + self.A2 @ self.E - self.b)) / (self.beta * self.eta2)
        E = self.f2_prox(W2, self.beta * self.eta2)

        # update lamda
        con1 = self.A1 @ Z + self.A2 @ E - self.b
        self.lamda += self.beta * con1
        if norm(con1) / norm(self.b) < self.eps1:
            flag1 = False

        # update beta
        con2 = self.beta * max(np.sqrt(self.eta1) * norm(Z - self.Z), np.sqrt(self.eta2) * norm(E - self.E)) / norm(self.b)
        if con2 < self.eps2:
            rho = self.rho0
        else:
            rho = 1
        self.beta = min(self.beta_max, rho * self.beta)
        if con2 < self.eps2:
            flag2 = False
        
        self.E = E.copy()
        self.Z = Z.copy()

        return flag1 or flag2

    def opt(self, Z0, E0, lamda, F, beta0, beta_max, rho0, eps1, eps2):
        self.Z, self.E, self.D, self.lamda = Z0, E0, F.D, lamda
        self.f1, self.f2, self.A1, self.A2, self.b, self.eta1, self.eta2, self.f1_prox, self.f2_prox = F.f1, F.f2, F.A1, F.A2, F.b, F.eta1, F.eta2, F.f1_prox, F.f2_prox
        self.beta, self.beta_max, self.rho0, self.eps1, self.eps2 = beta0, beta_max, rho0, eps1, eps2
        self.bs = []
        self.fs = []
        self.gap = []
        flag = True
        step = 0
        while flag:
            flag = self._step()
            step += 1 
        self.L = len(self.fs)

    def plot_fi(self):
        plt.figure()
        if not hasattr(self, "pstar"):
            self.pstar = self.fs[-1]
        plt.plot(np.linspace(1, self.L, self.L), np.log10(abs(np.array(self.fs) - self.pstar) + 1e-16))
        plt.savefig("fi.png")
        plt.show()

    def plot_bi(self):
        plt.figure()
        plt.plot(np.linspace(1, self.L, self.L), self.bs)
        plt.savefig("bi.png")
        plt.show()

    def plot_gi(self):
        plt.figure()
        plt.plot(np.linspace(1, self.L, self.L), np.log10(np.array(self.gap)+1e-16))
        plt.savefig("gi.png")
        plt.show()
        
def test_F(Z0, E0, D, mu, lamda, beta0, beta_max, rho0, eps1, eps2):
    opt = Opt()
    func = F(D, mu)
    opt.opt(Z0, E0, lamda, func, beta0, beta_max, rho0, eps1, eps2)
    ans = opt.fs[:]
    print(ans[-1])
    opt.plot_fi()
    opt.plot_bi()
    opt.plot_gi()
    del opt
    return ans

if __name__ == "__main__":
    D = randn(200, 300)
    E0 = np.zeros((200, 300))
    Z0 = np.zeros((300, 300))
    lamda = np.zeros((201, 300))
    mu = 1
    beta0 = 1e-4
    beta_max = 1e4
    rho0 = 1.5
    eps1 = 1e-5
    eps2 = 1e-5
    test_F(Z0, E0, D, mu, lamda, beta0, beta_max, rho0, eps1, eps2)