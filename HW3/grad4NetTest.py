from grad4Net import NetGraph
import json
import numpy as np


with open("HW3\grad4Net.json") as f:
    config = json.load(f)
    G = NetGraph(config)
    loss1 = G.loss(G.forward())
    G.backward()
    grad = G.finalGrad()
    t = 0.00001
    p = G.perturb(t)
    loss2 = G.loss(G.forward())
    print(np.dot(grad, p))
    print((loss2.value - loss1.value) / t)
    print("如果这两个数相差很小，说明测试成功")
