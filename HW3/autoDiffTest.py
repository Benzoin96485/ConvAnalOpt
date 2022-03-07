from autoDiff import ComputeGraph
import json
import numpy as np


def configPerturb(config, t):
    xVec = np.array(list(config["var"].values()))
    dxVec = np.random.rand(*xVec.shape)
    xVec2 = xVec + t * dxVec
    for i, varName in enumerate(config["var"].keys()):
        config["var"][varName] = xVec2[i]
    return dxVec


with open("HW3/autoDiff_config.json") as f:
    config = json.load(f)
    G = ComputeGraph()
    value1 = G.forward(config)
    G.backward()
    grad = G.finalGrad()
    t = 0.0001
    d = configPerturb(config, t)
    value2 = G.forward(config)
    dy_eval = (value2 - value1) / t
    dy_calc = (np.dot(grad, d))
    print(dy_eval - dy_calc)
    