'''
HW 12-2: Damped Newton and Gauss-Newton Algorithm for Rosenbrock’s function
'''
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from time import time


class Newton:
    def __init__(self, x0=np.array([-2, 2]), alpha=0.5, beta=0.5, eta=1e-5):
        self.x0 = x0
        self.alpha = alpha
        self.beta = beta
        self.x = x0
        self.eta2 = eta ** 2
        self.fs = []
        self.sls = []
        self.L = 0
        self.ts = []

    def f(self, x: np.array):
        # the i-th row of A is a_i^T
        return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

    def df(self, x: np.array):
        df0 = 400 * (x[0] ** 2 - x[1]) * x[0] + 2 * (x[0] - 1)
        df1 = 200 * (x[1] - x[0] ** 2)
        return np.array([df0, df1])

    def ddf(self, x: np.array):
        ddf00 = 800 * x[0] ** 2 + 400 * (x[0] ** 2 - x[1]) + 2
        ddf01 = -400 * x[0]
        ddf11 = 200
        return np.array([
            [ddf00, ddf01],
            [ddf01, ddf11]
        ])

    def rf(self, x: np.array):
        r0 = 10 * np.sqrt(2) * (x[1] - x[0] ** 2)
        r1 = np.sqrt(2) * (1 - x[0])
        return np.array([r0, r1])

    def Jf(self, x: np.array):
        J00 = -20 * np.sqrt(2) * x[0]
        J01 = 10 * np.sqrt(2)
        J10 = -np.sqrt(2)
        J11 = 0
        return np.array([
            [J00, J01],
            [J10, J11]
        ])

    def _step_damped(self):
        df = self.df(self.x)
        dx = inv(self.ddf(self.x)) @ df
        df2 = np.dot(df, df)
        if df2 < self.eta2:
            return True
        fx = self.f(self.x)
        t = 1
        while self.f(self.x + t * dx) > fx - self.alpha * t * df2:
            t *= self.beta
        self.x = self.x + t * dx
        self.fs.append(fx)
        self.ts.append(time() - self.t0)
        return False

    def _step_gauss(self):
        fx = self.f(self.x)
        self.fs.append(fx)
        self.ts.append(time() - self.t0)
        df = self.df(self.x)
        df2 = np.dot(df, df)
        J = self.Jf(self.x)
        JT = J.transpose()
        dx = -inv(JT @ J) @ (JT @ self.rf(self.x))
        self.x = self.x + dx
        if df2 < self.eta2:
            return True
        return False
    
    def opt(self, method="damped"):
        self.t0 = time()
        self._step = getattr(self, f"_step_{method}")
        while True:
            if self._step():
                break
        self.L = len(self.fs)

    def plot_fi(self):
        plt.figure()
        plt.plot(range(self.L), self.fs)
        plt.show()

    def plot_sli(self):
        plt.figure()
        plt.plot(range(self.L), self.sls)
        plt.show()

def plot_converge():
    np.random.seed(114514)
    nt = Newton()
    nt.opt(method="gauss")
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    ax = plt.gca()
    ax.set_xlabel("t")
    ax.set_ylabel("$\mathrm{lg}(f(x)-p^{*})$")
    plt.plot(nt.ts, np.log(np.array(nt.fs)))
    plt.subplot(1, 2, 2)
    ax = plt.gca()
    ax.set_xlabel("i")
    ax.set_ylabel("$\mathrm{lg}(f(x)-p^{*})$")
    plt.plot(range(nt.L), np.log(np.array(nt.fs)))
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_converge()