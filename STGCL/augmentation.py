#-----------------------------------
# augementation.py
# contain augementation methods
# ref: STGCL
#-----------------------------------

from pygments import highlight
import torch

import torch.nn.functional as F
import numpy as np

from random import random
from scipy.fftpack import fft, dct, idct

EIS = 20

def edgeMasking(input, r):
  """
    input: [207 * 207] numpy
    output: [207 * 207]
    delete the entries of adjacency matrix
    share within batch
  """
  nodeCount = input[0].size
  #generate random matrix M ~ U(0, 1)  
  randomMatrix = torch.rand(nodeCount, nodeCount)
  for i in range(nodeCount):
    for j in range(nodeCount):
      if randomMatrix[i][j] < r:
        input[i, j] = 0

def inputMasking(input, r):
  """
    Input Masking
    deleting entries of original input feature
    input: [batch, sequence, node, feature] e.g.[64, 12, 207, 2]
    output: [64, 12, 207, 2]
  """
  #keep original data unchanged
  output = input.clone().detach()
  batch, sequenceCnt, nodeCount, featuresCnt = input.shape[0],input.shape[1],input.shape[2],input.shape[3]
  randomMatrix = torch.rand(nodeCount)

  for b in range(batch):
    for i in range(nodeCount):
      if randomMatrix[i] < r:
        output[b,:,i, :] = torch.zeros(sequenceCnt, featuresCnt)
    
    return output

#temporal shifting
def temporalShift(input, r):
  """
    Temporal Shifting
    shift the data along the time axis
    P(t-S:t) = aX(t-S:t) + (1-a)(X(t-S+1: t) + 0)
    input: [batch, sequence, node, feature] e.g.[64, 12, 207, 2]
    output: tensor [64, 12, 207, 2]
  """
  a = r + (1-r) * random()

  p1d =  (0, 0, 0, 0, 0, 1, 0, 0)
  extended = input.clone().detach()
  extended = F.pad(extended[:, 1:, :, :], p1d, "constant", 0)
  return a * input + (1-a) * extended
  
#input smoothing
def inputSmooth(x, y, r):
  """
    Input Smoothing
    x: [batch, sequence, node, feature] e.g.[64, 12, 207, 2]
    y: [batch, sequence, node, feature] e.g.[64, 12, 207, 2]
    output: numpy [batch, sequence, node, feature] e.g.[64, 24, 207, 2]
  """
  input = np.concatenate((x, y), axis=1)
  dctX = dct(input)
  sequence, nodeCnt = input.shape[1], input.shape[2]

  randomMatrix = r + (1 - r) * np.random.rand(sequence - EIS, nodeCnt)
  #(4, 207)
  lowF = dctX[:, :EIS, :, :]
  highF = dctX[:, EIS:, :, :]
  #(64,4,207,2)
  sHighF = np.einsum('ij,kijl->kijl', randomMatrix, highF)

  output = np.concatenate((lowF, sHighF), axis=1)

  # return idct(output)
  return idct(output)/4


