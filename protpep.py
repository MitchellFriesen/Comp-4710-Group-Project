#!/usr/bin/env python2.7
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

with open("Dataset.txt", "r") as infile:
  Xfull = []
  Yfull = []

  possibleAA = ['G','P','A','V','L','I','M','C','F','Y','W','H','K','R','Q','N','E','D','S','T']

  cont  = True
  while(cont):
    line = infile.readline()
    if(line and line[0] == '>'):
      protAA = []
      # load residues:
      for i, x in enumerate(infile.readline()):
        aaVector = [0] * len(possibleAA)
        if(x in possibleAA):
          aaVector[possibleAA.index(x)] = 1
          protAA.append(aaVector)
      Xfull.append(protAA)
      
      # load peptide binding possibilities (0 or 1):
      protBind = []
      for y in infile.readline():
        if str.isdigit(y):
          protBind.append(float(y))
      Yfull.append(protBind)
    else:
      cont = False

  X = np.array(Xfull)
  Y = np.array(Yfull)
