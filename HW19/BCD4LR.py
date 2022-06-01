'''
HW 19-3: Block Coordinate Descent for low-rank matrix completion
'''
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import seed, randn, randint, shuffle
from numpy.linalg import norm, inv, svd
import random

random.seed(114515)
seed(114515)

class Opt:
    def __init__(self, D, P):
        self.D = D
        self.P = P

    def f(self):
        return norm(self.U @ self.V.T - self.A) / 2

    def _step(self):
        f = self.f()
        d = norm(self.D - self.A)
        print(f"f:{f}\td:{d}")
        self.fs.append(f)
        self.ds.append(d)
        A = self.U @ self.V.T
        self.A = A + self.P * (self.D - A)    
        self.U = self.A @ self.V @ inv(self.V.T @ self.V)
        self.V = (inv(self.U.T @ self.U) @ self.U.T @ self.A).T
        return True
        

    def opt(self, U0, V0, A0):
        self.U = U0
        self.V = V0
        self.A = A0
        self.fs = []
        self.ds = []
        flag = True
        step = 0
        while flag and step < 10000:
            flag = self._step()
            step += 1 
        self.L = len(self.fs)

    def plot_fi(self):
        plt.figure()
        plt.plot(np.linspace(1, self.L, self.L), np.log10(np.array(self.fs)))
        plt.ylabel(r"$\log f$")
        plt.xlabel("iter")
        plt.savefig(f"HW19/19_3_fi.png")
        plt.show()

    def plot_di(self):
        plt.figure()
        plt.plot(np.linspace(1, self.L, self.L), np.log10(np.array(self.ds)))
        plt.ylabel(r"$\log \|\mathbf{D}-\mathbf{A}\|_{F}$")
        plt.xlabel("iter")
        plt.savefig(f"HW19/19_3_di.png")
        plt.show()


def test_F(U0, V0, A0, D, P):
    opt = Opt(D, P)
    opt.opt(U0, V0, A0)
    opt.plot_fi()
    opt.plot_di()
    del opt
    
if __name__ == "__main__":
    U0 = randn(200, 5)
    V0 = randn(300, 5)
    A0 = randn(200, 300)
    D = randn(200, 5) @ randn(5, 300)
    P = np.zeros(A0.size)
    indices = np.arange(A0.size)
    shuffle(indices)
    P[indices[:int(0.1 * A0.size)]] = 1
    P = P.reshape(A0.shape)
    test_F(U0, V0, A0, D, P)

