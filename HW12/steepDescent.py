'''
HW 12-1: Steepest Descent Algorithm for function in HW 11-3
'''
import numpy as np
import matplotlib.pyplot as plt


class SD:
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
        df = self.df(self.x)
        dx = -np.sign(df) * np.sum(np.abs(df))
        df2 = np.dot(df, df)
        if df2 < self.eta2:
            return True
        fx = self.f(self.x)
        t = 1
        while self.f(self.x + t * dx) > fx + self.alpha * t * np.dot(df, dx):
            t *= self.beta
        self.x = self.x + t * dx
        self.fs.append(fx)
        self.sls.append(t * np.sqrt(np.dot(dx, dx)))
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
    avgLs = np.zeros(81)
    for seed in range(114514, 114524):
        np.random.seed(seed)
        A = np.random.randn(3,3)
        x0 = np.random.randn(3)
        betas = np.linspace(0.1, 0.9, 81)
        Ls = []
        for beta in betas:
            print(f"beta: {beta}")
            sd = SD(A, x0, alpha=0.5, beta=beta, eta=1e-6)
            sd.opt()
            Ls.append(sd.L)
        avgLs += np.array(Ls)
        plt.plot(betas, Ls)
    plt.plot(betas, avgLs / 10, linewidth=3)
    best_beta = betas[list(avgLs).index(min(avgLs))]
    print(best_beta)
    plt.savefig("HW12/L_beta.png")
    plt.show()
    plt.figure()
    plt.xlabel(r"$\alpha$")
    plt.ylabel("L")
    avgLs = np.zeros(81)
    for seed in range(114514, 114524):
        np.random.seed(seed)
        A = np.random.randn(3,3)
        x0 = np.random.randn(3)
        alphas = np.linspace(0.1, 0.9, 81)
        Ls = []
        for alpha in alphas:
            print(f"alpha: {alpha}")
            sd = SD(A, x0, alpha=alpha, beta=best_beta, eta=1e-6)
            sd.opt()
            Ls.append(sd.L)
        avgLs += np.array(Ls)
        plt.plot(alphas, Ls)
    plt.plot(alphas, avgLs / 10, linewidth=3)
    best_alpha = alphas[list(avgLs).index(min(avgLs))]
    print(best_alpha)
    plt.savefig("HW12/L_alpha.png")
    plt.show()
    print(f"best: alpha: {best_alpha}, beta: {best_beta}")
    return best_alpha, best_beta

def plot_converge(alpha, beta):
    np.random.seed(114514)
    A = np.random.randn(3,3)
    x0 = np.random.randn(3)
    sd = SD(A, x0, alpha, beta, eta=1e-6)
    sd.opt()
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    ax = plt.gca()
    ax.set_xlabel("i")
    ax.set_ylabel(r"$\|t\Delta x\|_{2}$")
    plt.plot(range(sd.L), np.log10(np.array(sd.sls)))
    plt.subplot(1, 2, 2)
    ax = plt.gca()
    ax.set_xlabel("i")
    ax.set_ylabel("$\mathrm{lg}(f(x)-p^{*})$")
    plt.plot(range(sd.L), np.log10(np.array(sd.fs) - 6))
    plt.tight_layout()
    plt.savefig("HW12/converge.png")
    plt.show()

if __name__ == "__main__":
    plot_converge(*search_param())
