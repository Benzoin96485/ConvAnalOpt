'''
HW 20: Several randomized algorithm:
    - Stochastic gradient descent
    - Stochastic variance reduced gradient
    - Momentum
    - Nesterov accelerated gradient
    - AdaGrad
    - AdaDelta
    - Adaptive moment estimation
    - Nesterov accelerated adaptive moment estimation
    - Randomized coordinate descent (gamma)
for large-scale logistic regression problem
'''

import numpy as np
import matplotlib.pyplot as plt
from time import time
from numpy.random import seed, randn, randint, RandomState, choice
from numpy.linalg import norm
import random
random.seed(114515)
seed(114515)

class Opt():
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.N = X.shape[1]
        self.M = X.shape[0]
        self.rowbeta = norm(X, axis=1) ** 2 / 4 / self.N
        self.beta = max(norm(X, axis=0)) ** 2 / 4 / self.N
        self.alpha = min(norm(X, axis=0)) ** 2 / 4 / self.N
        self.k = int(20 * self.beta / self.alpha + 0.5)

    def f(self, w):
        return np.sum(np.log(1 + np.exp(-(w @ self.X) * self.y))) / self.N

    def df(self, w):
        z = -(w.T @ self.X) * self.y
        g = -self.y * (1 - 1 / (np.exp(z) + 1))
        return self.X @ g / self.N

    def dfi(self, w, i):
        xi = self.X[:,i]
        z = -(w.T @ xi) * self.y[i]
        g = -self.y[i] * (1 - 1 / (np.exp(z) + 1))
        return g * xi / self.N

    def dfj(self, w, j):
        z = -(w.T @ self.X) * self.y
        g = -self.y * (1 - 1 / (np.exp(z) + 1))
        return self.X[j] @ g / self.N
    
    def _step_SGD(self, step, i):
        self.w -= self.dfi(self.w, i) * self.eta
        return True

    def _step_SVRG(self, step, i):
        f = self.f(self.w)
        if step % self.k == 0:
            self.w = np.average(self.wtmp, axis=0)
            self.gtmp = self.df(self.w)
        self.wtmp[step % self.k] = self.w.copy()
        self.w -= (self.dfi(self.w, i) - self.dfi(self.wtmp[0], i) + self.gtmp) * self.eta
        
    def _step_momentum(self, step, i):
        v = self.gamma * self.v + self.dfi(self.w, i) * self.eta
        self.w -= v
        self.v = v.copy()

    def _step_NAG(self, step, i):
        v = self.gamma * self.v + self.dfi(self.w - self.gamma * self.v, i) * self.eta
        self.w -= v
        self.v = v.copy()

    def _step_AdaGrad(self, step, i):
        gi = self.dfi(self.w, i)
        self.gis[i] += np.dot(gi, gi)
        self.w -= self.eta_AdaGrad / np.sqrt(1e-8 + self.gis[i]) * gi

    def _step_AdaDelta(self, step, i):
        gi = self.dfi(self.w, i)
        self.Eg2 = self.gamma * self.Eg2 + (1 - self.gamma) * gi ** 2
        dw = np.sqrt(self.dw2 + 1e-8) / np.sqrt(self.Eg2 + 1e-8) * gi
        self.dw2 = self.gamma * self.dw2 + (1 - self.gamma) * dw ** 2
        self.w -= dw

    def _step_Adam(self, step, i):
        gi = self.dfi(self.w, i)
        self.Eg = self.beta1 * self.Eg + (1 - self.beta1) * gi
        self.Eg2 = self.beta2 * self.Eg2 + (1 - self.beta2) * gi ** 2
        m = self.Eg / (1 - self.beta1 ** (step + 1))
        v = self.Eg2 / (1 - self.beta2 ** (step + 1))
        self.w -= self.eta_Adam / (np.sqrt(v) + 1e-8) * m
        pass

    def _step_Nadam(self, step, i):
        gi = self.dfi(self.w, i)
        self.Eg = self.beta1 * self.Eg + (1 - self.beta1) * gi
        self.Eg2 = self.beta2 * self.Eg2 + (1 - self.beta2) * gi ** 2
        m = self.Eg / (1 - self.beta1 ** (step + 1))
        v = self.Eg2 / (1 - self.beta2 ** (step + 1))
        self.w -= self.eta_Adam / (np.sqrt(v) + 1e-8) * (self.beta1 * m + (1 - self.beta1) * gi / (1 - self.beta1 ** (step + 1)))

    def _step_RCD(self, step, i):
        gj = self.dfj(self.w, i)
        self.w[i] -= gj / self.rowbeta[i]

    def opt(self, w0, method, gamma=0.9, beta1=0.9, beta2=0.999):
        rnd = RandomState(114514)
        self.method = method
        self.fs = []
        self.ts = []
        self.fstar = 0
        self.gtmp = None
        self.gis = np.zeros(self.N)
        self.Eg = np.zeros(self.M)
        self.Eg2 = np.zeros(self.M)
        self.dw2 = np.zeros(self.M)
        self.gamma = gamma
        self.beta1 = beta1
        self.beta2 = beta2
        self.w = w0
        self.eta = self.beta
        self.eta_AdaGrad = 0.001
        self.eta_Adam = 0.0001
        self.v = np.zeros_like(w0)
        self.pbeta = self.rowbeta ** self.gamma / np.sum(self.rowbeta ** self.gamma)
        self.wtmp = np.stack([w0] * self.k, axis=0)
        self._step = getattr(self, f"_step_{method}")
        flag = True
        step = 0
        self.t0 = time()
        while not self.ts or self.ts[-1] < 60:
            if method == "RCD":
                i = choice(self.M, p=self.pbeta)
            else:
                i = rnd.randint(0, high=self.N)
            f = self.f(self.w)
            self.fstar = (self.fstar * step + f) / (step + 1)
            self._step(step, i)
            self.fs.append(self.fstar)
            self.ts.append(time() - self.t0)
            # if step % 1 == 0:
            #     print(self.fstar)
            step += 1
        self.L = len(self.fs)

    def plot_ft(self): 
        plt.plot(self.ts, self.fs, label=f"{self.method}")

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
    test_F(X, y, w0, "SGD")
    test_F(X, y, w0, "SVRG")
    test_F(X, y, w0, "Adam")
    test_F(X, y, w0, "AdaGrad")
    test_F(X, y, w0, "AdaDelta")
    test_F(X, y, w0, "RCD")
    test_F(X, y, w0, "NAG")
    test_F(X, y, w0, "momentum")
    test_F(X, y, w0, "Nadam")
    plt.xlabel("t/s")
    plt.ylabel("f")
    plt.legend()
    plt.show()


    
    