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
