'''
HW 14-4: Example for majorization minimization method
'''
import numpy as np
import matplotlib.pyplot as plt
import time
from numpy.linalg import inv

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

    def df(self, x):
        w = self.A.transpose() @ (self.A @ x - self.b)
        for i in range(w.shape[0]):
            if x[i] == 0:
                if abs(w[i]) <= self.lamda:
                    w[i] = 0
                elif w[i] > self.lamda:
                    w[i] = w[i] - self.lamda
                elif w[i] < -self.lamda:
                    w[i] = w[i] + self.lamda
            else:
                w[i] += np.sign(x[i])
        return w

    def df2(self, x):
        return self.A.transpose() @ (self.A @ x - self.b) + self.lamda * (1 /np.abs(self.x)) * x

    def _step(self, method=1):
        f = self.f(self.x)
        print(f)
        self.fs.append(f)
        self.ts.append(time.time() - self.t0)
        if method == 1:
            alpha = 1
            w = self.A.transpose() @ (self.A @ self.x - self.b)
            while True:
                x = np.zeros_like(self.x)
                for i in range(self.x.shape[0]):
                    xpos = self.x[i] - alpha * w[i] - alpha * self.lamda
                    xneg = self.x[i] - alpha * w[i] + alpha * self.lamda
                    if xpos > 0:
                        x[i] = xpos
                    elif xneg < 0:
                        x[i] = xneg
                    else:
                        x[i] = 0
                if self.f(x) < self.f(self.x):
                    break
                alpha *= self.mu
                #if alpha < 1e-15:
                    #return False
            g = self.df(x)
            if np.dot(g, g) < self.eta2:
                self.x = x
                return False
            self.x = x 
        if method == 2:
            x = inv(self.A.transpose() @ self.A + self.lamda * np.diag(1 / np.abs(self.x))) @ (self.A.transpose() @ self.b)
            g = self.df(x)
            if np.dot(g, g) < self.eta2:
                self.x = x
                return False
            self.x = x
        return True
    
    def opt(self, method=1):
        flag = True
        self.t0 = time.time()
        while flag: 
            flag = self._step(method)
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
    print(opt.A, opt.b)
    opt.opt(method=2)
    print(opt.x)
    print(opt.ts[-1])
    print(opt.L)
    opt.plot_ft()
    opt.plot_fi()

if __name__ == "__main__":
    test()