'''
HW 14-4: Example for majorization minimization method
'''
import numpy as np
import matplotlib.pyplot as plt
import time
from numpy.random import rand

class majMin:
    def __init__(self, A=None, b=None, lamda=1, x0=None, seed=114514, mu=0.5, eta=1e-5):
        np.random.seed(seed)
        self.A = A if A else np.random.randn(3, 3)
        self.b = b if b else np.random.randn(3)
        self.lamda = lamda
        self.x = x0 if x0 else np.random.randn(3)
        self.mu = mu
        self.fs = []
        self.ts = []
        self.eta2 = eta ** 2

    def f(self, x):
        q = self.A @ x - self.b
        return np.dot(q, q) / 2 + self.lamda * np.sum(np.abs(x))

    def df1(self, x):
        return self.A.transpose() @ (self.A @ x - self.b)

    def _bktk_lsearch(self, d):
        alpha = 1
        while self.f(self.x + alpha * d) >= self.f(self.x) and alpha > 1e-10:
            alpha *= self.mu
        return alpha

    def _step(self, method=1):
        f = self.f(self.x)
        print(f)
        self.fs.append(f)
        self.ts.append(time.time() - self.t0)
        if method == 1:
            d = -self.df1(self.x)
        if np.dot(d, d) < self.eta2:
            return False
    
        alpha = self._bktk_lsearch(d)
        dx = alpha * d
        self.x += dx
        return True
    
    def opt(self):
        flag = True
        self.t0 = time.time()
        while flag: 
            flag = self._step()
        self.L = len(self.fs)
    
    def plot_fi(self):
        plt.figure()
        plt.plot(range(self.L), np.array(self.fs))
        plt.show()

    def plot_ft(self):
        plt.figure()
        plt.plot(self.ts, np.array(self.fs))
        plt.show()

def test():
    opt = majMin()
    opt.opt()
    print(opt.x)
    opt.plot_ft()
    opt.plot_fi()

if __name__ == "__main__":
    test()