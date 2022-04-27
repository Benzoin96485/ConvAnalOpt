'''
HW 14-2: BFGS Quasi-Newton Algorithm
'''
import numpy as np
import matplotlib.pyplot as plt
import time
from numpy.linalg import eigh


class F5:
    pstar = 0

    def f(x):
        return (x[0] - 3) ** 2 + 7 * (x[1] - x[0] ** 2) ** 2 + 9 * (x[2] - x[0] - x[1] ** 2) ** 2

    def df(x):
        df0 = 2 * (x[0] - 3) + 28 * (x[0] ** 2 - x[1]) * x[0] + 18 * (x[0] + x[1] ** 2 - x[2])
        df1 = 14 * (x[1] - x[0] ** 2) + 36 * (x[1] ** 2 + x[0] - x[2]) * x[1]
        df2 = 18 * (x[2] - x[0] - x[1] ** 2)
        return np.array([df0, df1, df2])


class F:
    def __init__(self, a=1):
        self.pstar = 0
        self.a = a

    def f(self, x):
        xc = x.reshape(-1, 2).copy()
        return np.sum(self.a * (xc[:,1] - xc[:,0] ** 2) ** 2 + (1 - xc[:,0]) ** 2)

    def df(self, x):
        xc = x.reshape(-1, 2).copy()
        xc1 = 2 * self.a * (xc[:,1] - xc[:,0] ** 2)
        xc0 = 4 * self.a * (xc[:,0] ** 2 - xc[:,1]) * xc[:,0] + 2 * (xc[:,0] - 1)
        xc[:,0] = xc0
        xc[:,1] = xc1
        return xc.reshape(-1)


class BFGS:
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
        while self.f(self.x + t * d) > self.f(self.x) + t * self.alpha * np.dot(g, d):
            t *= self.beta
        return t

    def _lsearch(self, d, f, g, c1=0.25, c2=0.75):
        tmin, tmax, t = 0, 1, 0.5

        # 二分搜索
        while tmax - tmin:
            x1 = self.x + t * d
            if self.f(x1) <= f + c1 * t * np.dot(g, d):
                if np.dot(self.df(x1), d) >= c2 * np.dot(g, d):
                    break
                else:
                    tmin = t
            else:
                tmax = t
            t = (tmin + tmax) / 2

        return t

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
        #alpha = self._bktk_lsearch(d, self.g)
        alpha = self._lsearch(d, f, self.g)
        dx = alpha * d
        self.x += dx
        g = self.df(self.x)
        dg = g - self.g
        Hdg = self.H @ dg
        dgdx = np.dot(dg, dx)
        Hdgdx = np.outer(Hdg, dx)
        self.H = self.H + (1 + np.dot(dg, Hdg) / dgdx) / dgdx * np.outer(dx, dx) - (Hdgdx + Hdgdx.transpose()) / dgdx
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

def testF5():
    bgfs = BFGS(F=F5)
    bgfs.opt(x0 = np.array([0., 0., 0.]), H0=np.eye(3), trackH=True)
    print(bgfs.x)
    bgfs.plot_ft()
    bgfs.plot_fi()
    bgfs.plot_Hi()
    print(min(bgfs.Hs))

def testF():
    bgfs = BFGS(F=F(1))
    bgfs.opt(x0 = np.array([-1.]*2000), H0=np.eye(2000), trackH=True)
    print(bgfs.x)
    bgfs.plot_ft()
    bgfs.plot_fi()
    bgfs.plot_Hi()
    print(min(bgfs.Hs))

if __name__ == "__main__":
    testF()