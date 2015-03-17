from pybrain.structure           import FeedForwardNetwork, FullConnection, LinearLayer, SigmoidLayer
from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from pylab                       import ion, ioff, figure, draw, contourf, clf, show, hold, plot
from scipy                       import diag, arange, meshgrid, where
from numpy.random                import multivariate_normal
import numpy as np

# from pybrain.tools.shortcuts import buildNetwork

net = FeedForwardNetwork(bias = True)

# Creating layers
inLayer = LinearLayer(1, name="input layer")
hiddenLayer = SigmoidLayer(2, name="hidden layer")
outLayer = LinearLayer(1, name="out layer")

# Creating connections
in_to_hidden = FullConnection(inLayer, hiddenLayer)
hidden_to_out = FullConnection(hiddenLayer, outLayer)

# Connecting everything
net.addInputModule(inLayer)
net.addModule(hiddenLayer)
net.addOutputModule(outLayer)
net.addConnection(in_to_hidden)
net.addConnection(hidden_to_out)

# Activate
net.sortModules()




# means = [(-1,0),(2,4),(3,1)]
# cov = [diag([1,1]), diag([0.5,1.2]), diag([1.5,0.7])]
# alldata = ClassificationDataSet(2, 1, nb_classes=3)
# for n in xrange(400):
    # for klass in range(3):
        # input = multivariate_normal(means[klass],cov[klass])
        # print input
        # alldata.addSample(input, [klass])

# tstdata, trndata = alldata.splitWithProportion( 0.25 )

# print tstdata

from parser import fromFile
trainraw = fromFile("sincTrain25.dt")
trndata = ClassificationDataSet(1,1)
for inp,out in trainraw:
    trndata.addSample(inp, out) #out*outlol)

testraw = fromFile("sincValidate10.dt")
tstdata = ClassificationDataSet(1,1)
for inp in testraw:
    tstdata.addSample(inp, 0)# out*outlol)

print tstdata

trndata._convertToOneOfMany( )
tstdata._convertToOneOfMany( )

print "Number of training patterns: ", len(trndata)
print "Input and output dimensions: ", trndata.indim, trndata.outdim
print "First sample (input, target, class):"
print trndata['input'][0], trndata['target'][0], trndata['class'][0]



trainer = BackpropTrainer( net, dataset=trndata, momentum=0.1, verbose=True, weightdecay=0.01)

ticks = arange(-3.,6.,0.2)
X, Y = meshgrid(ticks, ticks)
# need column vectors in dataset, not arrays
griddata = ClassificationDataSet(2,1, nb_classes=3)
for i in xrange(X.size):
        griddata.addSample([X.ravel()[i],Y.ravel()[i]], [0])
griddata._convertToOneOfMany()  # this is still needed to make the fnn feel comfy
