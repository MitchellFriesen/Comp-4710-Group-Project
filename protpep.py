#!/usr/bin/env python2.7
import numpy as np
from keras.layers import Convolution1D
from keras.layers import Flatten
from keras.layers import MaxPooling1D
from keras.layers.core import Dense, Activation
from keras.models import Sequential

possibleAA = ['G', 'P', 'A', 'V', 'L', 'I', 'M', 'C', 'F', 'Y', 'W', 'H', 'K', 'R', 'Q', 'N', 'E', 'D', 'S', 'T']

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

def getData(path, possibleAA, validation):
    Xfull = []
    Yfull = []
    with open(path, "r") as infile:
        cont = True
        while (cont):
            line = infile.readline()
            if line and line[0] == '>':
                protAA = []

                # load residues
                for i, x in enumerate(infile.readline()):
                    aaVector = [0] * len(possibleAA)
                    if x in possibleAA:
                        aaVector[possibleAA.index(x)] = 1
                        protAA.append(aaVector)
                Xfull.append(protAA)

                # load peptide binding possibilities (0 or 1)
                protBind = []
                for y in infile.readline():
                    if str.isdigit(y):
                        protBind.append(float(y))
                Yfull.append(protBind)
                if validation == True:
                    infile.readline()
            else:
                cont = False

        # create input matrix
        X = np.zeros((len(Xfull), 256, len(possibleAA)))
        for prot in range(X.shape[0]):
            protRange = range(min(256, len(Xfull[prot][:])))
            for protAA in protRange:
                for aaNum in range(len(possibleAA)):
                    X[prot][protAA][aaNum] = float(Xfull[prot][protAA][aaNum])


        # create desired output matrix
        for y in Yfull:
            for i in range(256 - len(y)):
                y.append(float(0))
            y = y[:256]
        Y = np.zeros((len(Yfull), 256))
        for y in range(len(Yfull)):
            for posAA in range(256):
                Y[y][posAA] = Yfull[y][posAA]

    return (X, Y)

X, Y = getData("Dataset.txt", possibleAA, False)
print "Got input data with shape: ", X.shape
print "Got training output data with shape:", Y.shape

model = Sequential()
model.add(Convolution1D(64, 15, input_shape=(256, len(possibleAA))))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_length=2))
model.add(Flatten())
model.add(Dense(500, init='normal', activation='relu'))
model.add(Dense(100, init='normal', activation='relu'))
model.add(Activation('relu'))
model.add(Dense(256))
model.add(Activation('sigmoid'))

print "Compiling..."
model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])
print "Compiled"

print "Fitting..."
model.fit(X, Y, nb_epoch=50, batch_size=10)
scores = model.evaluate(X, Y)
print "Fitting complete"

print "Performance:"
print("\n%s: %.2f%%\n" % (model.metrics_names[1], scores[1] * 100))

print "Validating using proteins that were actually use to train with (Todo: do real cross-validation, testing)..."
Xval, Yval = getData("TestSet_Prob.txt", possibleAA, True)
predict = model.predict(Xval)
print "Validation complete"

sum = 0.0
count = 0
sumz = 0.0
countz = 0
for yval, p in zip(Yval, predict):
    for yvalAA, pAA in zip(yval, p):
        if yvalAA == 1:
            sum += pAA
            count += 1
        else:
            sumz += pAA
            countz += 1
print "Validation results:"
print "  Overall sensitivity is: ", (sum / count)
print "  Overall specificity is: ", (sumz / countz)

print "  Expected binding sites for a sample protein:"
print Yval[0]
print "  Correct binding sites for sample protein:"
print(predict[0])
