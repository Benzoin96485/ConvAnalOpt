'''
HW 13-3: Conjugate Gradient Algorithm for Extended Rosenbrock’s function
'''
import numpy as np
import matplotlib.pyplot as plt
import time


class CG:
    def __init__(self, x0, a=1, alpha=0.5, beta=0.9, n=100, eta=1e-5):
        self.a = a # free param in extended Rosenbrock function
        self.alpha = alpha # alpha param in backtrack line search
        self.beta = beta # beta param in backtrack line search
        self.n = n # dimension of the vector x
        self.g = None # gradient
        self.d = None # direction
        self.x = x0 # initial x
        self.fs = [] # values of f
        self.ts = [] # time elapsed at steps
        self.eta2 = eta ** 2 # eta param of termination
        pass

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

    def _bktk_lsearch(self, d, g):
        t = 1
        while self.f(self.x + t * d) > self.f(self.x) + t * self.alpha * np.dot(g, d):
            t *= self.beta
        return t * self.alpha
        
    def _step(self, redirect=False, formula="HS"):
        f = self.f(self.x)
        #print(f)
        self.fs.append(f)
        self.ts.append(time.time() - self.time0)
        g = self.df(self.x)
        if np.dot(g, g) < self.eta2:
            return False
        if redirect:
            d = -g
        else:
            beta = 1
            if formula == "HS": # Hestenes-Stiefel
                beta = np.dot(g, g - self.g) / np.dot(self.d, g - self.g)
            elif formula == "PR": # Polak-Ribiere
                beta = np.dot(g, g - self.g) / np.dot(self.g, self.g)
            elif formula == "FR": # Fletcher-Reeves
                beta = np.dot(g, g) / np.dot(self.g, self.g)
            d = beta * self.d - g
        alpha = self._bktk_lsearch(d, g)
        self.x += alpha * d
        self.d = d.copy()
        self.g = g.copy()
        return True

    def opt(self, formula="HS"):
        step = 0
        flag = True
        self.time0 = time.time()
        while flag:
            flag = self._step(redirect=not (step % self.n), formula=formula)
            step += 1
        self.L = len(self.fs)

    def plot_fi(self):
        plt.figure()
        plt.plot(range(self.L), np.log10(self.fs))
        plt.show()

    def plot_ft(self):
        plt.figure()
        plt.plot(self.ts, np.log10(self.fs))
        plt.show()

    

if __name__ == "__main__":
    x0 = np.array([-1.0] * 100)
    cg = CG(x0, alpha=0.8, beta=0.2)
    cg.opt("FR")
    print(cg.ts[-1])
    print(cg.L)
    cg.plot_ft()
    cg.plot_fi()
    