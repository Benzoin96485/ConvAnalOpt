# HW3-3: a program for computing the gradient of error function V with respect
# to the weights of a neural network with arbitrary topology

from turtle import forward
import numpy as np
from autoDiff import ComputeGraph, Node, relu
np.random.seed(114514)


class Conv1dLayer:
    def __init__(self, graph, inChannel, padding, kernelWidth, outChannel, res, **kwargs):
        self.res = res
        self.graph = graph
        self.inChannel = inChannel
        self.outChannel = outChannel
        self.padding = padding
        self.kernelWidth = kernelWidth
        self.numParam = inChannel * kernelWidth * outChannel
        self.nodes = {(i, j): [Node(value=np.random.rand()) for k in range(kernelWidth)] for i in range(inChannel) for j in range(outChannel)}

    def _signIn(self):
        for nodeList in self.nodes.values():
            for node in nodeList:
                self.graph.addNodes(node, start=True)

    def _resSum(self, thisChannel, newChannel):
        out = []
        for i in range(len(thisChannel)):
            out.append(thisChannel[i] + newChannel[i])
        return out

    def forward(self, inputV):
        self._signIn()
        output = []
        for i in range(self.inChannel):
            thisChannel = self.padding * [0] + inputV[i] + self.padding * [0]
            thisWidth = self.padding * 2 + len(inputV[i])
            for j in range(self.outChannel):
                newChannel = []
                for l in range(thisWidth - self.kernelWidth + 1):
                    convSum = 0
                    for k in range(self.kernelWidth):
                        convSum = convSum + self.nodes[(i, j)][k] * thisChannel[l + k]
                    newChannel.append(convSum)
                if self.res:
                    newChannel = self._resSum(inputV[i], newChannel)
                output.append(newChannel)
        return output


class ReLULayer:
    def __init__(self, inShape, **kwargs):
        self.inShape = inShape

    def forward(self, inputV):
        if len(self.inShape) == 2:
            output = [[relu(inputV[i][j]) for j in range(self.inShape[1])] for i in range(self.inShape[0])]
        elif len(self.inShape) == 1:
            output = [relu(inputV[j]) for j in range(self.inShape[0])]
        return output


class LinearLayer:
    def __init__(self, graph, inWidth, outWidth, **kwargs):
        self.graph = graph
        self.nodesW = {(i, j): Node(np.random.rand()) for i in range(inWidth) for j in range(outWidth)}
        self.nodesB = [Node(np.random.rand()) for j in range(outWidth)]
        self.inWidth = inWidth
        self.outWidth = outWidth

    def _signIn(self):
        for node in self.nodesW.values():
            self.graph.addNodes(node, start=True)
        for node in self.nodesB:
            self.graph.addNodes(node, start=True)

    def _flatten(self, inputV):
        out = []
        for channel in inputV:
            for node in channel:
                out.append(node)
        return out

    def forward(self, inputV):
        self._signIn()
        inputV = self._flatten(inputV)
        output = []
        for j in range(self.outWidth):
            linearSum = 0
            for i in range(self.inWidth):
                linearSum = linearSum + self.nodesW[(i, j)] * inputV[i]
            linearSum = linearSum + self.nodesB[j]
            output.append(linearSum)
        return output


class NetGraph(ComputeGraph):
    def __init__(self, config):
        super().__init__()
        self.layerList = []
        self.initValue = config["initValue"]
        for layer in config["layers"]:
            if layer["type"] == "conv1d":
                self.layerList.append(Conv1dLayer(self, **layer))
            elif layer["type"] == "linear":
                self.layerList.append(LinearLayer(self, **layer))
            elif layer["type"] == "relu":
                self.layerList.append(ReLULayer(**layer))
        self.lossFunc = config["lossFunc"]
        self.label = config["label"]
        
    def forward(self):
        result = self.initValue
        for layer in self.layerList:
            result = layer.forward(result)
        return result
    
    def loss(self, result):
        if self.lossFunc == "l2":
            lossSum = 0
            for i, x in enumerate(result):
                lossSum = lossSum + (x + -self.label[i]) * (x + -self.label[i])
            return lossSum

    def perturb(self, t):
        l = len(self.startNodes)
        p = np.random.rand(l)
        for i in range(l):
            self.startNodes[i].value += p[i] * t
        return p

                

            
