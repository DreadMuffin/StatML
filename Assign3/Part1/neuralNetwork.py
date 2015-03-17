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
    error = (1.0/(2.0 * len(target))) * sum(difpow)
    print error
    return error

def estPartials(ws, epsilon):
    retarray = np.empty([len(ws),len(ws)])
    for i in range(len(ws)):
        ei = np.array([0.0] * len(ws))
        ei[i] = 1.0
        retarray[i,:] = ((ws + epsilon * ei) - ws) / epsilon
    return retarray


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
    # layers.append(la.Layer("Hid Layer2", learnConst))
    layers.append(la.Layer("Out Layer", learnConst, momentum))
    layers[2].isOutput()
    say("----------------------------------------------------------")
    say("Done creating layers! \n")

    say("Adding nodes to layers")
    say("----------------------------------------------------------")
    layers[0].addNeuron(ne.Neuron(linear, "Inp neuron", False))
    layers[1].addNeuron(ne.Neuron(nonLinear, "Hid neuron11", False))
    layers[1].addNeuron(ne.Neuron(nonLinear, "Hid neuron12", False))
    # layers[2].addNeuron(ne.Neuron(nonLinear, "Hid neuron21"))
    # layers[2].addNeuron(ne.Neuron(nonLinear, "Hid neuron22"))
    # layers[2].addNeuron(ne.Neuron(nonLinear, "Hid neuron23"))
    # layers[2].addNeuron(ne.Neuron(nonLinear, "Hid neuron24"))
    # layers[2].addNeuron(ne.Neuron(nonLinear, "Hid neuron25"))
    # layers[2].addNeuron(ne.Neuron(nonLinear, "Hid neuron26"))
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
networkTalk = False
# networkTalk = True

# Network constants
learnConst = 0.000001
momentum = 0.00001

inputLayer, outputLayer= createNetwork(learnConst, momentum)
errors = []

# print "Training!"

inputVector      = []
outputCalculated = []
outputTarget     = []

training = fromFile("sincTrain25.dt")

# Loop
loops = 8
for _ in range(loops):

    results = []
    targets = []

    for i in range(len(training)):

        (input, targetOut) = training[i]
        inputLayer.getNeurons()[1].setValue(input) 
        inputLayer.calculate()
        output = outputLayer.getNeurons()[0].getValue()

        targets.append(targetOut)
        results.append(output)

    error = meanSquareError(targets, results)
    print error
    print outputLayer.getNeurons()[0].getValue()
    print targetOut
    outputLayer.backpropogate([error])

    ws = []
    for w in inputLayer.getWeights([]):
        ws.append(w[2])

    print estPartials(np.array([ws]),0.1)
    break


test = fromFile("sincValidate10.dt")
for i in range(len(test)):
    (input, targetOut) = test[i]
    inputLayer.getNeurons()[1].setValue(input) 
    inputLayer.calculate()
    output = outputLayer.getNeurons()[0].getValue()

    # print "Target -> " + str(targetOut)
    # print "Predicted -> " + str(output)

    inputVector.append(input)
    outputCalculated.append(output)
    outputTarget.append(targetOut)

inputVectorSorted, outputTarget= (list(t) for t in zip(*sorted(zip(inputVector, outputTarget))))
plt.plot(np.array(inputVector),np.array(outputCalculated), "ro")
plt.plot(np.array(inputVectorSorted),np.array(outputTarget), "b-")
plt.show()
