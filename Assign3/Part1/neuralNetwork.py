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
    dif = map(sub, target, actual)
    difpow = map(power, dif)
    error = (1.0/(2.0 * len(target))) * sum(difpow)
    return error

def estPartials(eT):
    epsilon = 0.00000001

    llen = len(layers) - 1
    wlen = len(inputLayer.getWeights([]))

    ws = [[] for i in range(llen)]

    for i in range(llen):
        ws[i] = layers[i].getLayerWeights()

    wValues = []
    for l in ws:
        for w in l:
            wValues.append(w[2])

    retarray = [0.0] * wlen

    for i in range(wlen):
        ei = np.array([0.0] * wlen)
        ei[i] = 1.0

        print wValues
        print epsilon
        print ei

        Eeps = wValues + epsilon * ei

        altWeights = [[] for r in range(llen)]

        a = 0

        print ws
        for l in range(llen):
            for w in ws[l]:
                altWeights[l].append((w[0],w[1],Eeps[a],0))
                a += 1

        for l in range(len(ws)):
            print inputLayer.getWeights([])[l][2]

        for k in range(len(test)):
            (input, targetOut) = test[k]
            inputLayer.setWeights(altWeights)
            inputLayer.calculate()
            output = outputLayer.getNeurons()[0].getValue()

            tmptarget = targetOut
            tmpresult = output

        altMSE = meanSquareError([tmptarget], [tmpresult])
        retarray[i] = (altMSE - eT)



    return retarray


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
    layers[1].addNeuron(ne.Neuron(nonLinear, "Hid neuron11", False))
    layers[1].addNeuron(ne.Neuron(nonLinear, "Hid neuron12", False))
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




# def normalize(dif, input, output)

# debugging
layers = []

# Network constants
learnConst = 0.050000
momentum = 0.05000

# print "Training!"


training = fromFile("sincTrain25.dt")
test = fromFile("sincValidate10.dt")

# Loop
loops = 4000
networkTalk = False
# networkTalk = True
inputLayer, outputLayer= createNetwork(learnConst, momentum)
errorsT = []
errorsTest = []
for i in range(loops):

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

    results = []
    targets = []

    # Test
    for i in range(len(test)):
        (input, targetOut) = test[i]
        inputLayer.getNeurons()[1].setValue(input)
        inputLayer.calculate()
        output = outputLayer.getNeurons()[0].getValue()

        targets.append(targetOut)
        results.append(output)

    errorTest = meanSquareError(targets, results)
    errorsTest.append(abs(errorTest))


print estPartials(errorTest)

plt.plot(np.array(errorsT), "b-", label = "Training set")
plt.plot(np.array(errorsTest), "r-", label = "Test set")
plt.legend()
plt.xlabel("Number of runs")
plt.ylabel("Bached Mean Square Error")
plt.show()
