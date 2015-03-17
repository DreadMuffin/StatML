class Neuron:
    talk = False
    name = "unnamed"
    isBias = False

    value = None

    transformFunc = None

    # reciever contains the neurons this neuron send information to (does not
    # include weigh)
    sendTo = None

    # reciever contains the neurons this neuron recieves information from
    # includes weigh)
    recieveFrom = None

    # A list that is dynamicly build up as parts of the equation is given
    calculation = None

    error = None

    def __init__(self, func, name, isBias):
        self.calculation = [] 
        self.sendTo = []
        self.recieveFrom = []
        self.name = name
        self.transformFunc = func
        self.isBias = isBias
        # self.say("Hello world!")

    def recieveWeigh(self, sender, weigh):
        self.say("got a weigh " + str(weigh) + " from " + sender.getName())
        self.calculation.append((sender, weigh))

    def calculateAndSend(self):
        self.calculateNewValue()
        self.sendAll() 

    def calculateNewValue(self):
        if self.calculation != []:
            self.newLine()
            self.say("Calculating!")
            # self.say(str(self.calculation))
            newValue = 0
            for (sender, weigh) in self.calculation:
                newValue += sender.getValue() * weigh
                self.say("    found weight: " + str(weigh) + " and value: " + str(sender.getValue()) + ". Calculated value: " + str(sender.getValue() * weigh))
            self.value = newValue/len(self.calculation)
            self.say("calculated value " + str(self.value))
        self.calculation = []

    # def calcError(self, otherError, weight):
        # self.error = otherError * weight
        # self.say("I calculated my error to " + str(self.error))

    def getError(self):
        return self.error

    def setError(self, error):
        self.say("I updated my error to " + str(error))
        self.error = error

    def addRecieveFrom(self, neigh):
        # self.say("got a new node named \"" + neigh.getName() + "\" listening to it!")
        self.recieveFrom.append((neigh, 1))

    def addSendTo(self, neigh):
        self.say("got a new node named \"" + neigh.getName() + "\" sending to it!")
        self.sendTo.append((neigh, 1))

    def transferFromNeuron(self, sender, x, weigh):
        self.say("got a value from \"" + sender.getName() + "\" saying " + str(x))
        self.value = self.transformFunc(x)
        # self.sendAll()

    def sendAll(self):
        if self.sendTo == []:
            self.newLine()
            self.say("I have no next layer. So outputting! -> " + str(self.value))
        else:
            for (neighbor, weigh) in self.sendTo:
                neighbor.transferFromNeuron(self, self.value, weigh)

    def updateWeigh(self, x):
        # Do something to update weights
        for (neighbor, weigh) in self.recieveFrom:
            neighbor.updateWeigh(x)

    def getValue(self):
        # self.say("Im outputting: " + str(self.value))
        return self.value

    def setValue(self, value):
        self.say("Setting my value to " + str(value))
        self.value = value

    def say(self, message):
        if self.talk:
            print "    " + self.name + ": " + message

    def newLine(self):
        if self.talk:
            print ""

    def getName(self):
        return self.name
