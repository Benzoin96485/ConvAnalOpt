'''
HW 13-4,5: DFP Quasi-Newton Algorithm
'''
import numpy as np
import matplotlib.pyplot as plt
import time
from numpy.linalg import eigh

class F4:
    pstar = -0.75

    def f(x):
        return x[0] ** 4 / 4 + x[1] ** 2 / 2 - x[0] * x[1] + x[0] - x[1]

    def df(x):
        df0 = x[0] ** 3 - x[1] + 1
        df1 = x[1] - x[0] - 1
        return np.array([df0, df1])

class F5:
    pstar = 0

    def f(x):
        return (x[0] - 3) ** 2 + 7 * (x[1] - x[0] ** 2) ** 2 + 9 * (x[2] - x[0] - x[1] ** 2) ** 2

    def df(x):
        df0 = 2 * (x[0] - 3) + 28 * (x[0] ** 2 - x[1]) * x[0] + 18 * (x[0] + x[1] ** 2 - x[2])
        df1 = 14 * (x[1] - x[0] ** 2) + 36 * (x[1] ** 2 + x[0] - x[2]) * x[1]
        df2 = 18 * (x[2] - x[0] - x[1] ** 2)
        return np.array([df0, df1, df2])

class DFP:
    def __init__(self, F, alpha=0.5, beta=0.5, eta=1e-5):
        self.x = 0 # initial x
        self.H = None # initial approximated Hessian
        self.g = None # gradient
        self.alpha = alpha # alpha param in line search
        self.beta = beta # beta param in line search
        self.f = F.f
        self.df = F.df
        self.pstar = F.pstar
        self.fs = []
        self.ts = []
        self.Hs = []
        self.eta2 = eta ** 2
        self.time0 = time.time()
    
    def _bktk_lsearch(self, d, g):
        t = 1
        while self.f(self.x + t * d) > self.f(self.x) + t * self.alpha * np.dot(g, d) and t > 1e-8:
            t *= self.beta
        return t * self.alpha

    def _trackH(self):
        self.Hs.append(min(eigh(self.H)[0]))

    def _step(self, init=False, trackH=False):
        f = self.f(self.x)
        print(f)
        self.fs.append(f)
        self.ts.append(time.time() - self.time0)
        if init:
            self.g = self.df(self.x)
        if np.dot(self.g, self.g) < self.eta2:
            return False
        d = -self.H @ self.g
        alpha = self._bktk_lsearch(d, self.g)
        dx = alpha * d
        self.x += dx
        g = self.df(self.x)
        dg = g - self.g
        Hdg = self.H @ dg
        self.H = self.H + np.outer(dx, dx) / np.dot(dx, dg) - np.outer(Hdg, Hdg) / np.dot(dg, Hdg)
        if trackH:
            self._trackH()
        self.g = g.copy()
        return True
    
    def opt(self, x0, H0, trackH):
        self.x = x0
        self.H = H0
        if trackH:
            self._trackH()
        step = 0
        flag = True
        while flag: 
            flag = self._step(step == 0, trackH)
            step += 1
        self.L = len(self.fs)
    
    def plot_fi(self):
        plt.figure()
        plt.plot(range(self.L), np.log10(np.array(self.fs) - self.pstar))
        plt.show()

    def plot_ft(self):
        plt.figure()
        plt.plot(self.ts, np.log10(np.array(self.fs) - self.pstar))
        plt.show()

    def plot_Hi(self):
        plt.figure()
        plt.plot(range(self.L), self.Hs)
        plt.show()

    
def testF4():
    dfp = DFP(F=F4)
    dfp.opt(x0 = np.array([1.5, 1.]), H0=np.eye(2))
    print(dfp.x)
    dfp.plot_ft()
    dfp.plot_fi()

def testF5():
    dfp = DFP(F=F5)
    dfp.opt(x0 = np.array([0., 0., 0.]), H0=np.eye(3), trackH=True)
    print(dfp.x)
    dfp.plot_ft()
    dfp.plot_fi()
    dfp.plot_Hi()
    print(min(dfp.Hs))

if __name__ == "__main__":
    testF5()
    