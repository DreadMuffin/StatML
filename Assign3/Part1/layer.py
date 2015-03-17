import random
import math
import neuron as ne

class Layer:
    talk = False
    name = "unnamed"
    neurons = None

    weights = None
    nextLayer = None
    prevLayer = None

    learnConst = None
    mom = None


# Creation

    def linear(self, x):
        return x

    def __init__(self, name, learnConst, momentum):
        self.mom = momentum
        self.learnConst = learnConst
        self.weights = []
        self.name = name
        self.neurons = []
        bias = ne.Neuron(self.linear, name + "bias", True)
        bias.setValue(1)
        self.neurons.append(bias)
        self.say("Hello World!")

    def addNeuron(self, neuron):
        self.say("got something named \"" + neuron.getName() + "\"")
        self.neurons.append(neuron)

    def getNeurons(self):
        return self.neurons

    def sendToLayer(self, layer):
        self.nextLayer = layer
        layer.prevLayer = self
        self.say("sending to " + layer.getName())
        for sender in self.neurons:
            for reciever in layer.getNeurons():
                if not reciever.isBias:
                    # self.weights.append((sender, reciever, random.uniform(0,1), 0))
                    self.weights.append((sender, reciever, 1, 0))
                    sender.addSendTo(reciever)
                    reciever.addRecieveFrom(sender)

    def isOutput(self):
        self.say("Am output, removing bias")
        self.neurons = []

# Backpropogation

    def backpropogate(self, errors):
        # On every layer that is not output, we calculate the error
        if self.nextLayer is not None:
            errors = self.nextLayer.getErrors()
            # Calculates new error for each neuron
            for neuronThis in self.neurons:
                accError = 0
                for (neuronNext, _) in errors:
                    # Finds the weight between the 2 layers
                    weight = self.findWeight(neuronThis, neuronNext)
                    accError += neuronNext.getError() * weight 
                # self.say("The accumulated error for " + neuronThis.getName() + " is " + str(accError))
                newError = self.derivativeOfLinearFunc(neuronThis.getValue()) * accError
                neuronThis.setError(newError)

            for i in range(len(self.weights)):
                sender, reciever, weight, momentum = self.weights[i]
                newMomentum = self.learnConst * newError * sender.getValue()
                newWeight = weight + newMomentum + self.mom * momentum
                self.say("Updating " + sender.getName() + "'s weight from " + str(weight) + " to " + str(newWeight))
                self.weights[i] = (sender, reciever, newWeight, newMomentum)

        # only used in the output layer. Ensures the last layer does not find
        # an error nor calculate weights after it as it is given.
        else:
            for i in range(len(self.neurons)):
                self.neurons[i].setError(errors[i])

        # Recursive call on all previous layers
        if self.prevLayer is not None:
            self.prevLayer.backpropogate(0)

    def getErrors(self):
        errors = []
        for neuron in self.neurons:
            errors.append((neuron, neuron.getError()))
        return errors 

    def findWeight(self, sender1, reciever1):
        for (sender2, reciever2, weight, _) in self.weights:
            if sender2 == sender1 and reciever1 == reciever2:
                return weight
        return 0

    def derivativeOfLinearFunc(self, x):
        return 1.0/((1.0 + abs(x)) ** 2.0)

# Forward propogation

    def calculate(self):
        self.newLine()
        self.say("Asking my layer to calculate and send")
        if self.prevLayer is not None:
            for neuron in self.neurons:
                neuron.calculateAndSend()
        else:
            for neuron in self.neurons:
                neuron.sendAll()

        if self.nextLayer is not None:
            self.newLine()
            self.say("Sending weights to next layer")
            for (sender, reciever, weigh, _) in self.weights:
                reciever.recieveWeigh(sender, weigh)
            self.nextLayer.calculate()


# Debugging 

    def getName(self):
        return self.name

    def getWeights(self, retWeights):
        for sender, reciever, weight, _ in self.weights:
           retWeights.append((sender, reciever, weight, reciever.getValue()))
        if self.nextLayer is not None:
            return self.nextLayer.getWeights(retWeights)
        else:
            return retWeights

    def setWeights(self, weights):
        self.weights = weights

    def say(self, message):
        if self.talk:
            print self.name + ": " + message

    def newLine(self):
        if self.talk:
            print "\n"
