'''
HW3-4: An automatic differentiation program that works on a given expression
'''

import numpy as np
np.random.seed(1919810)


class ComputeGraph:
    def __init__(self):
        self.nodes = []
        self.startNodes = []
        self.nodeNum = 0
    
    def addNodes(self, node, start=False):
        node.graph = self
        node.index = self.nodeNum
        self.nodeNum += 1
        self.nodes.append(node)
        if start == True:
            self.startNodes.append(node)

    def forward(self, config):
        for varName, value in config["var"].items():
            if config["randomValue"]:
                value = np.random.rand()
                config["var"]["varName"] = value
            print(f"{varName} = {value}")
            exec(f"{varName} = Node(value={value})")
            self.addNodes(eval(f"{varName}"), start=True)
        return eval(config["exp"]).value
    

    def backward(self):
        for i in range(self.nodeNum):
            self.nodes[-i-1].backward(mid=i)
    
    def finalGrad(self):
        gradList = []
        for node in self.startNodes:
            gradList.append(node.grad)
        return np.array(gradList)


class Node:
    def __init__(self, value, preList=[], graph=None):
        self.preList = preList
        self.value = value
        self.graph = graph
        self.grad = 0
        self.index = -1

    def backward(self, mid=False):
        if not mid:
            self.grad = 1
        if self.preList:
            for nodeIndex, edgeValue in self.preList:
                self.graph.nodes[nodeIndex].grad += edgeValue * self.grad
        else:
            return

    def __add__(self, x):
        if type(x) == type(self):
            result = Node(value=self.value + x.value, preList=[(x.index, 1)])
        else:
            result = Node(value=self.value + x, preList=[])
        result.preList.append((self.index, 1))
        self.graph.addNodes(result)
        return result

    def __radd__(self, x):
        if type(x) == type(self):
            result = Node(value=self.value + x.value, preList=[(x.index, 1)])
        else:
            result = Node(value=self.value + x, preList=[])
        result.preList.append((self.index, 1))
        self.graph.addNodes(result)
        return result

    def __mul__(self, x):
        if type(x) == type(self):
            result = Node(value=self.value * x.value, preList=[(x.index, self.value), (self.index, x.value)])
        else:
            result = Node(value=self.value * x, preList=[(self.index, x)])
        self.graph.addNodes(result)
        return result

    def __rmul__(self, x):
        if type(x) == type(self):
            result = Node(value=self.value * x.value, preList=[(x.index, self.value), (self.index, x.value)])
        else:
            result = Node(value=self.value * x, preList=[(self.index, x)])
        self.graph.addNodes(result)
        return result


def sin(x):
    result = Node(value=np.sin(x.value), preList=[(x.index, np.cos(x.value))])
    x.graph.addNodes(result)
    return result
        

def cos(x):
    result = Node(value=np.cos(x.value), preList=[(x.index, -np.sin(x.value))])
    x.graph.addNodes(result)
    return result


def tan(x):
    result = Node(value=np.tan(x.value), preList=[(x.index, 1 / np.cos(x.value) ** 2)])
    x.graph.addNodes(result)
    return result


def exp(x):
    result = Node(value=np.exp(x.value), preList=[(x.index, np.exp(x.value))])
    x.graph.addNodes(result)
    return result


def log(x):
    result = Node(value=np.log(x.value), preList=[(x.index, 1 / x.value)])
    x.graph.addNodes(result)
    return result

def relu(x):
    if x.value > 0:
        result = Node(value=x.value, preList=[(x.index, 1)])
    else:
        result = Node(value=0, preList=[(x.index, 0)])
    x.graph.addNodes(result)
    return result


if __name__ == "__main__":
    pass
