'''
HW 14-3: L-BFGS Quasi-Newton Algorithm
'''
import numpy as np
import matplotlib.pyplot as plt
import time


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


class Memory:
    def __init__(self, m):
        self.m = m
        self.memory = [None for i in range(m)]

    def __getitem__(self, i):
        return self.memory[i % self.m]

    def __setitem__(self, i, v):
        self.memory[i % self.m] = v


class LBFGS:
    def __init__(self):
        pass

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


    def _step(self, k):
        f = self.f(self.x)
        # print(f)
        self.fs.append(f)
        self.ts.append(time.time() - self.t0)

        # 收敛准则
        g = self.df(self.x)
        if np.dot(g, g) < self.eta2:
            return False

        # 选择 H_k^0
        if k == 0:
            H0 = self.H0
        else:
            dg = g - self.g
            self.dg[k - 1] = dg
            dxdg = np.dot(self.dx[k - 1], dg)
            self.rho[k - 1] = 1 / dxdg
            gamma = dxdg / np.dot(dg, dg)
            H0 = gamma * np.eye(self.n)
        
        # 第一循环：计算 q
        q = g.copy()
        for i in range(k - 1, max(0, k-self.m) - 1, -1):
            alpha = self.rho[i] * np.dot(self.dx[i], q)
            self.alphas[i] = alpha
            q -= alpha * self.dg[i]
        
        # 第二循环：计算 p
        p = H0 @ q
        for i in range(max(0, k-self.m), k):
            beta = self.rho[i] * np.dot(self.dg[i], p)
            p += (self.alphas[i] - beta) * self.dx[i]

        # 更新
        d = -p
        t = self._lsearch(d, f, g, 0.25, 0.75)
        self.x += t * d
        self.dx[k] = t * d
        self.g = g.copy()
        return True

    def opt(self, F, x0, H0, m, eta):
        self.fs = []
        self.ts = []
        self.f = F.f
        self.df = F.df
        self.pstar = F.pstar
        self.x = x0
        self.n = x0.shape[0]
        self.H0 = H0
        self.m = m
        self.eta2 = eta ** 2
        self.dx = Memory(self.m)
        self.dg = Memory(self.m)
        self.rho = Memory(self.m)
        self.alphas = Memory(self.m)
        step = 0
        flag = True
        self.t0 = time.time()
        while flag: 
            flag = self._step(k=step)
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

def test_F():
    ms = range(3, 31)
    l_bgfs = LBFGS()
    Ls = []
    ts = []
    for m in ms:
        print(m)
        l_bgfs.opt(
            F=F(1),
            x0=np.array([-1.] * 2000),
            H0=np.eye(2000),
            m=m,
            eta=1e-5
        )
        Ls.append(l_bgfs.L)
        ts.append(l_bgfs.ts[-1])
    fig = plt.figure()
    plt.subplot(1, 2, 1)
    ax = plt.gca()
    ax.set_xlabel("$m$")
    ax.set_ylabel("$L$")
    plt.plot(ms, Ls)
    plt.subplot(1, 2, 2)
    ax = plt.gca()
    ax.set_xlabel("$m$")
    ax.set_ylabel("$t$")
    plt.plot(ms, ts)
    fig.tight_layout()
    plt.savefig("HW14/L-BFGS.png")
    plt.show()

if __name__ == "__main__":
    test_F()