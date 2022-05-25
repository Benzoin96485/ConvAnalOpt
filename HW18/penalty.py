'''
HW 18-1: compare absolute value penalty and Courant-Beltrami penalty for the least-square solution of underdetermined linear equations
'''
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy.random import seed, randn
import random
random.seed(114514)
seed(114514)

class Q:
    def __init__(self, A, b, penalty, gamma):
        self.penalty = penalty
        self.A = A
        self.b = b
        self.gamma = gamma

    def f(self, x):
        return np.dot(x, x) / 2

    def q(self, x):
        f = self.f(x)
        Axb = self.A @ x - self.b
        if self.penalty == "abs":
            return f + self.gamma * np.sum(np.abs(Axb))
        elif self.penalty == "CB":
            return f + self.gamma * np.sum(Axb ** 2)
    
    def dq(self, x):
        Axb = self.A @ x - self.b
        if self.penalty == "abs":
            subg = self.gamma * self.A.T @ np.sign(Axb)
            subg_b = np.abs(self.gamma * self.A.T @ np.ones_like(Axb))
            g = np.zeros_like(x)
            for i in range(x.shape[0]):
                if subg[i] == 0:
                    if abs(x[i]) <= subg_b[i]:
                        g[i] = 0
                    elif x[i] > subg_b[i]:
                        g[i] = x[i] - subg_b[i]
                    else:
                        g[i] = x[i] + subg_b[i]
                else:
                    g[i] = x[i] + subg[i]
            return g
        elif self.penalty == "CB":
            return x + self.gamma * 2 * self.A.T @ Axb

class Opt:
    def __init__(self):
        pass

    def _lsearch(self, x, d, f, g, func):
        t = 100
        while func(x + t * d) > f + self.alpha * t * np.dot(d, g):
            t *= self.beta
        return t

    def _step(self, step):
        f = self.f(self.x)
        g = self.df(self.x)
        d = -g

        #print(f)
        self.fs.append(self.f0(self.x))
        if np.dot(d, d) < self.eta2 or step > 5000:
            return False  
        
        alpha = self._lsearch(self.x, d, f, g, self.f)
        self.x += alpha * d
        return True

    def opt(self, Q, alpha, beta, eta):
        self.A = Q.A; self.b = Q.b
        self.x = randn(300)      
        self.f = Q.q; self.df = Q.dq; self.f0 = Q.f
        self.alpha = alpha; self.beta = beta; self.eta2 = eta ** 2 
        self.fs = []
        self.pstar = self.f(self.A.T @ (inv(self.A @ self.A.T) @ self.b))
        print(self.pstar)
        flag = True 
        step = 0
        while flag:
            flag = self._step(step)
            step += 1 
        self.L = len(self.fs)
        

    def plot_fi(self):
        plt.figure()
        plt.plot(np.linspace(1, self.L, self.L), np.log10(abs(np.array(self.fs) - self.pstar)))
        plt.show()

def test_F(A, b, penalty, gamma):
    opt = Opt()
    func = Q(A, b, penalty, gamma)
    opt.opt(Q=func, alpha=0.25, beta=0.75, eta=1e-5)
    ans = opt.fs[:]
    print(ans[-1])
    del opt
    return ans

if __name__ == "__main__":
    A = randn(200, 300)
    b = randn(200)
    f = lambda x: np.dot(x, x) / 2
    pstar = f(A.T @ (inv(A @ A.T) @ b))
    ans1 = test_F(A, b, "abs", 0.0001)
    ans2 = test_F(A, b, "abs", 0.1)
    ans3 = test_F(A, b, "CB", 0.0001)
    ans4 = test_F(A, b, "CB", 0.1)
    plt.figure()
    plt.plot(range(len(ans1)), np.log10(abs(np.array(ans1) - pstar) + 1e-16), label="abs, gamma=1e-4")
    plt.plot(range(len(ans2)), np.log10(abs(np.array(ans2) - pstar) + 1e-16), label="abs, gamma=0.1")
    plt.plot(range(len(ans3)), np.log10(abs(np.array(ans3) - pstar) + 1e-16), label="CB, gamma=1e-4")
    plt.plot(range(len(ans4)), np.log10(abs(np.array(ans4) - pstar) + 1e-16), label="CB, gamma=0.1")
    plt.legend()
    plt.savefig("HW18/penalty.png")


