'''
HW 19-2: Block Coordinate Descent for dictionary learning
'''
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import seed, randn, randint
from numpy.linalg import norm, inv, svd
import random
random.seed(114515)
seed(114515)

class Opt:
    def __init__(self, Y, lamda):
        self.Y = Y
        self.lamda = lamda
    
    def f(self, X, D):
        return norm(self.Y - D @ X) / 2 + self.lamda * norm(X, ord=1)

    def fDj(self, Dj):
        D = self.D.copy()
        D[:,self.j] = Dj
        return norm(self.Y - D @ self.X) / 2 + self.lamda * norm(X, ord=1)

    def fXj(self, Xj):
        X = self.X.copy()
        X[self.j,:] = Xj
        return norm(self.Y - self.D @ X) / 2

    def _lsearch(self, x, d, f, g, func):
        t = 1
        while func(x + t * d) > f + self.alpha * t * np.dot(d, g):
            t *= self.beta
        return t

    def thres(self, x0):
        x = x0.copy()
        for i in range(x.shape[0]):
            if x[i] > self.lamda:
                x[i] -= self.lamda
            elif x[i] < -self.lamda:
                x[i] += self.lamda
            else:
                x[i] = 0
        return x

    def _step(self):
        f = self.f(self.X, self.D)
        print(f)
        self.fs.append(f)
        
        for j in range(self.k):
            x = self.X[j,:]
            d = self.D[:,j]

            res = self.Y - self.D @ self.X + np.outer(d, x)
            U, sigma, Vt = svd(res)
            d = U[:,0]
            x = Vt[0,:] * sigma[0]
            x = self.thres(x)

            self.X[j,:] = x
            self.D[:,j] = d
        return True

    def opt(self, X0, D0, alpha, beta):
        self.fs = []
        self.X = X0
        self.D = D0
        self.k = D0.shape[1]
        self.alpha = alpha
        self.beta = beta
        flag = True
        step = 0
        while flag and step < 50:
            flag = self._step()
            step += 1 
        self.L = len(self.fs)

    def plot_fi(self):
        plt.figure()
        plt.plot(np.linspace(1, self.L, self.L), self.fs)
        plt.ylabel(r"$f(x)$")
        plt.xlabel("iter")
        plt.savefig(f"19_2_fi.png")
        plt.show()

def test_F_grad(Y, X0, D0, lamda): 
    opt = Opt(Y, lamda)
    opt.opt(X0, D0, alpha=0.25, beta=0.75)
    ans = opt.fs[:]
    print(ans[-1])
    opt.plot_fi()
    del opt
    return ans

if __name__ == "__main__":
    Y = randn(200, 500)
    X = randn(400, 500)
    D = randn(200, 400)
    for j in range(D.shape[1]):
        D[:,j] = D[:,j] / norm(D[:,j])
    lamda = 1
    test_F_grad(Y, X, D, lamda)