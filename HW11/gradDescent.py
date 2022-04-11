import numpy as np
import matplotlib.pyplot as plt


class GD:
    def __init__(self, A: np.array, x0: np.array, alpha: float, beta: float, eta: float):
        self.A = A
        self.x0 = x0
        self.alpha = alpha
        self.beta = beta
        self.x = x0
        self.eta2 = eta ** 2
        self.fs = []
        self.sls = []
        self.L = 0

    def f(self, x: np.array):
        # the i-th row of A is a_i^T
        return np.sum(np.exp(self.A @ x) + np.exp(-self.A @ x))

    def df(self, x: np.array):
        return (np.exp(self.A @ x) - np.exp(-self.A @ x)) @ self.A

    def _step(self):
        dx = -self.df(self.x)
        df2 = np.dot(dx, dx)
        if df2 < self.eta2:
            return True
        fx = self.f(self.x)
        t = 1
        while self.f(self.x + t * dx) > fx - self.alpha * t * df2:
            t *= self.beta
        self.x = self.x + t * dx
        self.fs.append(fx)
        self.sls.append(t * np.sqrt(df2))
        return False

    def opt(self):
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

def search_param():
    plt.figure()
    plt.xlabel(r"$\beta$")
    plt.ylabel("L")
    avgLs = np.zeros(99)
    for seed in range(114514, 114524):
        np.random.seed(seed)
        A = np.random.randn(3,3)
        x0 = np.random.randn(3)
        betas = np.linspace(0.01, 0.99, 99)
        Ls = []
        for beta in betas:
            gd = GD(A, x0, alpha=0.5, beta=beta, eta=1e-6)
            gd.opt()
            Ls.append(gd.L)
        avgLs += np.array(Ls)
        plt.plot(betas, Ls)
    plt.plot(betas, avgLs / 10, linewidth=3)
    print(betas[list(avgLs).index(min(avgLs))])
    plt.show()
    plt.figure()
    plt.xlabel(r"$\alpha$")
    plt.ylabel("L")
    avgLs = np.zeros(99)
    for seed in range(114514, 114524):
        np.random.seed(seed)
        A = np.random.randn(3,3)
        x0 = np.random.randn(3)
        alphas = np.linspace(0.01, 0.99, 99)
        Ls = []
        for alpha in alphas:
            gd = GD(A, x0, alpha=alpha, beta=0.9, eta=1e-6)
            gd.opt()
            Ls.append(gd.L)
        avgLs += np.array(Ls)
        plt.plot(alphas, Ls)
    plt.plot(alphas, avgLs / 10, linewidth=3)
    print(alphas[list(avgLs).index(min(avgLs))])
    plt.show()

def plot_converge(alpha, beta):
    np.random.seed(114514)
    A = np.random.randn(3,3)
    x0 = np.random.randn(3)
    gd = GD(A, x0, alpha, beta, eta=1e-6)
    gd.opt()
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    ax = plt.gca()
    ax.set_xlabel("i")
    ax.set_ylabel(r"$\|t\Delta x\|_{2}$")
    plt.plot(range(gd.L), np.log(np.array(gd.sls)))
    plt.subplot(1, 2, 2)
    ax = plt.gca()
    ax.set_xlabel("i")
    ax.set_ylabel("$\mathrm{lg}(f(x)-p^{*})$")
    plt.plot(range(gd.L), np.log(np.array(gd.fs) - 6))
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_converge(alpha=0.56, beta=0.85)