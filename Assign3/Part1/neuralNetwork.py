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
    # difpow = map(power, dif)
    error = (1.0/(2.0 * len(target))) * sum(dif)
    return error

def estPartials(eT):
    epsilon = 0.00000001

    llen = len(layers) - 1
    wlen = len(inputLayer.getWeights([]))

    ws = [[] for i in range(llen)]

    for i in range(llen):
        ws[i] = layers[i].getLayerWeights()

    wslens = []

    for w in ws:
        wslens.append(len(w))


    wValues = []
    for l in ws:
        for w in l:
            wValues.append(w[2])

    retarray = [0.0] * wlen

    for i in range(wlen):
        ei = np.array([0.0] * wlen)
        ei[i] = 1.0


        Eeps = wValues + epsilon * ei


        altWeights = [[] for r in range(llen)]

        a = 0

        for l in range(llen):
            for w in ws[l]:
                altWeights[l].append((w[0],w[1],Eeps[a],0.0))
                a += 1

        for k in range(len(test)):
            (input, targetOut) = test[k]
            for l in range(llen):

                layers[l].setWeights(altWeights[l])
            inputLayer.calculate()
            output = outputLayer.getNeurons()[0].getValue()

            tmptarget = targetOut
            tmpresult = output

        altMSE = meanSquareError([tmptarget], [tmpresult])
        retarray[i] = (altMSE - eT) / epsilon

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
    for i in range(20):
        layers[1].addNeuron(ne.Neuron(nonLinear, "Hid neuron" + str(i), False))
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

test = []
training = []
for x in np.arange(-20,20, 0.1):
    y = sinc(x)
    test.append((x,y))
    training.append((x,y))

training = fromFile("sincTrain25.dt")
test = fromFile("sincValidate10.dt")
# plt.plot(x,sinc(x))
# plt.show()

# Loop
loops = 10
networkTalk = False
# networkTalk = True

for learnConst, momentum, colour in [(0.5,0.5, "r"),(0.0001,0.0, "b"),(1,1, "g")]:

    layers = []
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

        errorTest = meanSquareError(targets, results)
        errorsTest.append(abs(errorTest))

    plt.plot(np.array(plotIn), np.array(results), colour + "-", label = "N2: (Learn, moment) = (" + str(learnConst) + ", " + str(momentum) + ")")

plt.plot(np.array(plotIn), np.array(targets), label = "Target")
plt.legend()
plt.show()

# print estPartials(errorTest)

#plt.plot(np.array(errorsT), "b-", label = "Training set")
#plt.plot(np.array(errorsTest), "r-", label = "Test set")
#plt.legend()
#plt.xlabel("Number of runs")
#plt.ylabel("Bached Mean Square Error")
#plt.show()
