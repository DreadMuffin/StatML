from time import sleep
from sklearn.metrics import mean_squared_error
import math
from parser import fromFile


import matplotlib.pyplot as plt
import numpy as np
import neuron as ne
import layer as la


# Helper functions
def sub(a,b):
    return a-b

def power(a):
    return a ** 2.0

def addConnection(a,b):
    a.sendToLayer(b)

def meanSquareError(target, actual):
    dif = map(sub, actual, target)
    difpow = map(power, dif)
    error = (1.0/(2.0 * len(target))) * sum(dif)
    return error 

def linearError(target, actual):
    dif = map(sub, actual, target)
    return sum(dif)/len(target)


def say(message):
    if networkTalk:
        print message

# Transfer funcions
def nonLinear(x):
    return x/(1.0 + abs(x))

def linear(x):
    return x


# Network API
def createNetwork(learnConst, momentum):
    ne.Neuron.talk = networkTalk
    la.Layer.talk = networkTalk
    say("Creating Layers")
    say("----------------------------------------------------------")
    layers.append(la.Layer("Int Layer", learnConst, momentum))
    layers.append(la.Layer("Hid Layer1", learnConst, momentum))
    # layers.append(la.Layer("Hid Layer2", learnConst, momentum))
    layers.append(la.Layer("Out Layer", learnConst, momentum))
    layers[2].isOutput()
    say("----------------------------------------------------------")
    say("Done creating layers! \n")

    say("Adding nodes to layers")
    say("----------------------------------------------------------")
    layers[0].addNeuron(ne.Neuron(linear, "Inp neuron", False))
    for i in range(2):
        layers[1].addNeuron(ne.Neuron(nonLinear, "Hid neuron" + str(i), False))
    # layers[1].addNeuron(ne.Neuron(nonLinear, "Hid neuron11", False))
    # layers[1].addNeuron(ne.Neuron(nonLinear, "Hid neuron12", False))
    # layers[2].addNeuron(ne.Neuron(nonLinear, "Hid neuron21", False))
    # layers[2].addNeuron(ne.Neuron(nonLinear, "Hid neuron22", False))
    # layers[2].addNeuron(ne.Neuron(nonLinear, "Hid neuron23", False))
    # layers[2].addNeuron(ne.Neuron(nonLinear, "Hid neuron24", False))
    # layers[2].addNeuron(ne.Neuron(nonLinear, "Hid neuron25", False))
    # layers[2].addNeuron(ne.Neuron(nonLinear, "Hid neuron26", False))
    layers[2].addNeuron(ne.Neuron(linear, "Out neuron", False))
    say("----------------------------------------------------------")
    say("Done adding! \n")

    say("Adding connections between layers")
    say("----------------------------------------------------------")
    addConnection(layers[0], layers[1])
    addConnection(layers[1], layers[2])
    # addConnection(layers[2], layers[3])
    say("----------------------------------------------------------")
    say("Done building! \n")
    return (layers[0], layers[2])

def broadcast(value, inputNeuron, outputNeuron):
    say("Broadcasting " + str(value))
    inputNeuron.broadcast(value)
    return outputNeuron.getValue() 

def sinc(x):
   return np.sinc(x) 

# debugging
layers = []

# Network constants
learnConst = 2.0000
momentum = 1.00000

# print "Training!"


test = []
training = []
for x in np.arange(-20,20, 00.1):
    y = sinc(x)
    test.append((x,y))
    training.append((x,y))

# training = fromFile("sincTrain25.dt")
# test = fromFile("sincValidate10.dt")
# plt.plot(x,sinc(x))
# plt.show()

# Loop
loops = 100
networkTalk = False
# networkTalk = True
inputLayer, outputLayer= createNetwork(learnConst, momentum)
errorsT = []
errorsTest = []
plotsOutTarget = []
plotsOutActual = []

print "Starting Loop"
for j in range(loops):

    results = []
    targets = []

    # Training 
    for i in range(len(training)):

        (input, targetOut) = training[i]
        inputLayer.getNeurons()[1].setValue(input) 
        inputLayer.calculate()
        output = outputLayer.getNeurons()[0].getValue()

        targets.append(targetOut)
        results.append(output)

    errorT = meanSquareError(targets, results)
    errorsT.append(abs(errorT))
    outputLayer.backpropogate([errorT])

    print "I: " + str(input)
    print "O: " + str(output)
    print "T: " + str(targetOut)
    print "E: " + str(errorT)

    results = []
    targets = []

    if (j % 250) == 0:
        print "Run: " + str(j)

results = []
targets = []
plotIn = []

# Test
for i in range(len(test)):
    (input, targetOut) = test[i]
    inputLayer.getNeurons()[1].setValue(input) 
    inputLayer.calculate()
    output = outputLayer.getNeurons()[0].getValue()

    targets.append(targetOut)
    results.append(output)

    plotIn.append(input)

print inputLayer.getWeights([])
errorTest = meanSquareError(targets, results)
errorsTest.append(abs(errorTest))

plt.plot(np.array(plotIn), np.array(results), "r-", label = "Calculated")
plt.plot(np.array(plotIn), np.array(targets), "b-", label = "Target")
plt.legend()
plt.show()

plt.plot(np.array(errorsT), "b-", label = "Training set")
plt.plot(np.array(errorsTest), "r-", label = "Test set")
plt.legend()
plt.xlabel("Number of runs")
plt.ylabel("Bached Mean Square Error")
plt.show()
