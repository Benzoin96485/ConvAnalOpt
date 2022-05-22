# ConvAnalOpt

北京大学信息科学技术学院智能科学与技术系核心课《凸分析与优化方法》编程作业仓库

- HW3
  - `autoDiff.py`，读入配置文件 `autoDiff.json`，利用自动微分计算给定函数的梯度，用 `autoDiffTest` 作测试
  - `grad4Net.py`，读入配置文件 `grad4Net.json`，计算给定拓扑的神经网络对参数的梯度，用 `grad4NetTest` 作测试
- HW11
  - `gradDescent.py`，在一个类双曲余弦函数上实现回溯线搜索（back-tracking line search）的梯度下降（gradient descent），同文件夹下的图片是一些可视化测试数据
- HW12
  - `steepDescent.py`，在 HW11 的函数上实现了基于无穷范数的最陡下降（steepest descent），同文件夹下的图片是一些可视化测试数据
  - `Newton.py`，在著名的 Rosenbrock 函数上实现了阻尼牛顿（damped Newton）法和高斯牛顿（Gauss-Newton）法
- HW13
  - `conjGrad.py`，在扩展的 Rosenbrock 函数上实现了共轭梯度（conjugate gradient）法
  - `DFP.py`，在一些多项式函数上实现了 DFP 拟牛顿算法
- HW14
  - `BFGS.py`，在最小二乘问题上实现了 BFGS 型拟牛顿算法
  - `L-BFGS.py`，在 Rosenbrock 函数上实现了有限内存的 BFGS 型拟牛顿算法
  - `majMin.py`，使用超优极小化（majorization minimization）方法优化带 L1 罚函数的最小二乘问题
- HW17
  - `compare.py`，比较了一些约束优化算法的性能
